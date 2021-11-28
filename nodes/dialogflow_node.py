#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A dialogflow integration for ROS

The node has support for events, intents, contexts, parameters and even dialogflow actions


License: BSD
  https://raw.githubusercontent.com/samiamlabs/dyno/master/LICENCE
"""
import uuid
import queue
import rospy
import random
import wave
import time
from collections import deque
from google.cloud import dialogflow
from google.protobuf import struct_pb2
from google.type import latlng_pb2
from google.api_core import exceptions

from std_msgs.msg import String, Bool, UInt16
from std_msgs.msg import Empty as EmptyMsg
from audio_common_msgs.msg import AudioData
from dialogflow_ros.msg import Response, Event, Context, Parameter
from qt_robot_interface.srv import audio_play, speech_config
from std_srvs.srv import Empty, EmptyResponse

class DialogflowNode:
    """ The dialogflow node """
    def __init__(self):
        rospy.init_node('dialogflow_node')

        self.project_id = "folke-jkih"
        self.session_id = str(uuid.uuid4())
        self.language = rospy.get_param('~default_language', 'sv')
        self.disable_audio = rospy.get_param('~disable_audio', False)
        
        time_before_start = rospy.get_param('~time_before_start', 0.8)
        self.save_audio_requests = rospy.get_param('~save_audio_requests', True)

        self.session_client = dialogflow.SessionsClient()
        
        self.query_params = dialogflow.QueryParameters(geo_location = latlng_pb2.LatLng(latitude=58.4106611, longitude=15.6198244),
                                                       contexts = [dialogflow.Context(lifespan_count=100,
                                                                                      name="projects/"+self.project_id+"/agent/sessions/"+self.session_id+"/contexts/linkoping"
                                                       )]
        )
        


        self.audio_chunk_queue = deque(maxlen=int(time_before_start * 31.25))# 16000/512 = 31.25,  # Times 7.8 since the data is sent in 7.8Hz (16000 / 2048)

        # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
        audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
        sample_rate_hertz = 16000
        self.audio_config = dialogflow.InputAudioConfig(
            audio_encoding=audio_encoding,
            language_code=self.language,
            sample_rate_hertz=sample_rate_hertz,
            single_utterance=True)

        self.query_result_pub = rospy.Publisher('response', Response, queue_size=2)
        self.query_text_pub = rospy.Publisher('query_text', String, queue_size=2)
        self.transcript_pub = rospy.Publisher('transcript', String, queue_size=2)
        self.fulfillment_pub = rospy.Publisher('fulfillment_text', String, queue_size=10)
        self.is_listening_pub = rospy.Publisher('is_listening', Bool, queue_size=2, latch=True)
        self.is_waiting_for_hot_word = rospy.Publisher('waiting_for_hot_word', Bool, queue_size=2, latch=True)
        self.volume = 0
        self.is_talking = False
        self.is_in_dialog = False
        self.detected_wake_word = False
        self.head_visible = False
        self.waiting_for_wake_word = False
        self.cancel_stream_intent = False
        self.skip_audio = False
        rospy.wait_for_service('/qt_robot/audio/play')
        self.audio_play_srv = rospy.ServiceProxy('/qt_robot/audio/play', audio_play)
        rospy.wait_for_service('/qt_robot/speech/config')
        self.speech_config_srv = rospy.ServiceProxy('/qt_robot/speech/config', speech_config)

        rospy.Subscriber('text', String, self.text_callback)
        rospy.Subscriber('is_talking', Bool, self.is_talking_callback)
        rospy.Subscriber('event', Event, self.event_callback)
        rospy.Subscriber('head_visible', Bool, self.head_visible_callback)
        rospy.Subscriber('detected_wake_word', String, self.detected_wake_word_callback)
        rospy.Subscriber('end_of_conversation', EmptyMsg, self.end_of_conversation_callback)

        if not self.disable_audio:
            rospy.Subscriber('sound', AudioData, self.audio_callback)
            rospy.Subscriber('volume', UInt16, self.volume_callback)

        self.list_intents_sevice = rospy.Service(
                'list_intents',
                Empty,
                self.handle_list_intents)

        self.list_context_sevice = rospy.Service(
                'list_context',
                Empty,
                self.handle_list_context)

        self.list_context_sevice = rospy.Service(
                'clear_context',
                Empty,
                self.handle_clear_context)

    def handle_list_intents(self, _):
        """ Prints all intents """
        intents_client = dialogflow.IntentsClient()
        parent = intents_client.project_agent_path(self.project_id)
        intents = intents_client.list_intents(parent)

        for intent in intents:
            rospy.loginfo('=' * 20)
            rospy.loginfo('Intent name: {}'.format(intent.name))
            rospy.loginfo('Intent display_name: {}'.format(intent.display_name))
            rospy.loginfo('Action: {}\n'.format(intent.action))
            rospy.loginfo('Root followup intent: {}'.format(
                intent.root_followup_intent_name))
            rospy.loginfo('Parent followup intent: {}\n'.format(
                intent.parent_followup_intent_name))

            rospy.loginfo('Input contexts:')
            for input_context_name in intent.input_context_names:
                rospy.loginfo('\tName: {}'.format(input_context_name))

            rospy.loginfo('Output contexts:')
            for output_context in intent.output_contexts:
                rospy.loginfo('\tName: {}'.format(output_context.name))
        return EmptyResponse()

    def handle_list_context(self, _):
        """ Prints the current contexts """
        contexts_client = dialogflow.ContextsClient()
        contexts = contexts_client.list_contexts(self.session)

        rospy.loginfo('Contexts for session {}:\n'.format(self.session))
        for context in contexts:
            rospy.loginfo('Context name: {}'.format(context.name))
            rospy.loginfo('Lifespan count: {}'.format(context.lifespan_count))
            rospy.loginfo('Fields:')
            for field, value in context.parameters.fields.items():
                if value.string_value:
                    rospy.loginfo('\t{}: {}'.format(field, value))
        return EmptyResponse()

    def handle_clear_context(self, _):
        """ Clear all current contexts """
        contexts_client = dialogflow.ContextsClient()
        contexts_client.delete_all_contexts(parent=self.session)
        return EmptyResponse()

    def is_talking_callback(self, msg):
        """ Callback for text input """
        self.is_talking = msg.data

    def detected_wake_word_callback(self, msg):
        """ Callback for text input """
        if self.waiting_for_wake_word:
            if msg.data == "swedish":
                self.language = 'sv'
                self.speech_config_srv("sv_SV", 500, 100)
            elif msg.data == "english":
                self.language = 'en'
                self.speech_config_srv("en_US", 150, 80)
            elif msg.data == "german":
                self.language = 'de'
                self.speech_config_srv("de_DE", 150, 100)
            elif msg.data == "chinese":
                self.language = 'zh-CN'
                self.speech_config_srv("zh_MA", 50, 100)
            else:
                rospy.logerr("Not valid language: " + msg.data)
                self.language = 'sv'
                self.speech_config_srv("sv_SV", 150, 100)
            self.audio_config.language_code = self.language
        

        self.detected_wake_word = True
        rospy.Timer(rospy.Duration(0.3), self.set_wake_word_false, oneshot=True)

    def end_of_conversation_callback(self, msg):
        self.end_of_dialog = True
        self.handle_clear_context(None)

    def set_wake_word_false(self, event):
        self.detected_wake_word = False

    def head_visible_callback(self, msg):
        """ Callback for text input """
        self.head_visible = msg.data
        #if not self.head_visible:
        #    self.cancel_stream_intent = True
        #else:
        #    self.cancel_stream_intent = False

    def text_callback(self, text_msg):
        """ Callback for text input """
        self.query_text_pub.publish(text_msg)
        self.end_of_dialog = False
        query_result = self.detect_intent_text(text_msg.data)
        if query_result.intent.end_interaction:
            self.end_of_dialog = True
        self.publish_response(query_result)

    def event_callback(self, event_msg):
        """ Callback for event input """
        rospy.loginfo("Publishing event %s", event_msg.name)
        self.end_of_dialog = False
        query_result = self.detect_intent_event(event_msg)
        if query_result.intent.end_interaction:
            self.end_of_dialog = True
        self.publish_response(query_result)

    def publish_response(self, query_result):
        """ Converts the dialogflow query result to the corresponding ros message """
        query_result_msg = Response()
        query_result_msg.project_id = self.project_id
        query_result_msg.query_text = query_result.query_text
        query_result_msg.intent_detection_confidence = query_result.intent_detection_confidence
        query_result_msg.language_code = query_result.language_code
        query_result_msg.intent.display_name = query_result.intent.display_name
        query_result_msg.intent.name = query_result.intent.name
        query_result_msg.intent.end_interaction = query_result.intent.end_interaction
        query_result_msg.action = query_result.action

        if not query_result.fulfillment_text:
            fulfillment_messages = query_result.fulfillment_messages
            if len(fulfillment_messages) > 0:
                query_result_msg.fulfillment_text = fulfillment_messages[0].text.text[0]
            else:
                rospy.logwarn("No fulfillment_text can be parsed")
        else:
            query_result_msg.fulfillment_text = query_result.fulfillment_text

        query_result_msg.parameters.extend(self.create_parameters(query_result.parameters))

        for ctx in query_result.output_contexts:
            name = ctx.name.split("/")[-1]
            if not name.endswith("__"):
                c_msg = Context()
                c_msg.name = name
                if ctx.parameters:
                    c_msg.parameters.extend(self.create_parameters(ctx.parameters))
                query_result_msg.output_contexts.append(c_msg)

        self.query_result_pub.publish(query_result_msg)
        self.fulfillment_pub.publish(query_result_msg.fulfillment_text)

    def create_parameters(self, params):
        """ Helper function for converting dialogflow messages to ROS counterparts """
        msg = []
        if not params:
            return msg
        for key in params:
            p_msg = Parameter()
            p_msg.key = key
            value = params.get(key)
            if isinstance(value, str):
                p_msg.value = [value]
            elif isinstance(value, float):
                p_msg.value = [str(value)]
            else:
                p_msg.value = value
            msg.append(p_msg)
        return msg

    def audio_callback(self, audio_chunk_msg):
        """ Callback for audio data """
        if self.skip_audio:
            return
        self.audio_chunk_queue.append(audio_chunk_msg.data)

    def volume_callback(self, msg):
        """ Callback for volume """
        self.volume = msg.data

    def detect_intent_text(self, text):
        """ Send text to dialogflow and publish response """
        self.cancel_stream_intent = True
        text_input = dialogflow.TextInput(text=text, language_code=self.language)
        query_input = dialogflow.QueryInput(text=text_input)
        response = self.session_client.detect_intent(session=self.session, query_input=query_input)

        rospy.loginfo('-' * 10 + " %s " + '-' * 10, self.project_id)
        rospy.loginfo('Query text: {}'.format(response.query_result.query_text))
        rospy.loginfo('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        rospy.loginfo('Fulfillment text: {}\n'.format(
            response.query_result.fulfillment_text))
        self.cancel_stream_intent = False
        return response.query_result

    def detect_intent_event(self, event_msg):
        """ Send event to dialogflow and publish response """
        self.cancel_stream_intent = True
        event_input = dialogflow.EventInput(language_code=self.language, name=event_msg.name)
        params = struct_pb2.Struct()
        for param in event_msg.parameters:
            params[param.key] = param.value
        event_input.parameters = params
        query_input = dialogflow.QueryInput(event=event_input)
        response = self.session_client.detect_intent(session=self.session, query_input=query_input)

        rospy.loginfo('-' * 10 + " %s " + '-' * 10, self.project_id)
        rospy.loginfo('Query text: {}'.format(response.query_result.query_text))
        rospy.loginfo('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        rospy.loginfo('Fulfillment text: {}\n'.format(
            response.query_result.fulfillment_text))
        self.cancel_stream_intent = False
        return response.query_result

    def detect_intent_stream(self):
        """ Send streaming audio to dialogflow and publish response """
        if self.disable_audio:
            return
        self.end_of_dialog = False
        requests = self.audio_stream_request_generator()
        responses = self.session_client.streaming_detect_intent(requests=requests)
        rospy.loginfo('=' * 10 + " %s " + '=' * 10, self.project_id)
        try:
            for response in responses:
                rospy.loginfo('Intermediate transcript: "{}".'.format(
                    response.recognition_result.transcript))
                response.recognition_result.transcript = response.recognition_result.transcript.replace("Lidköping", "Linköping")
                self.transcript_pub.publish(response.recognition_result.transcript)
        except exceptions.OutOfRange as exc:
            rospy.logerr("Dialogflow exception. Out of audio quota? "
                         "No internet connection (%s)", exc)
            return


        if self.cancel_stream_intent:
            return
        
        # pylint: disable=undefined-loop-variable
        query_result = response.query_result
        query_result.query_text = query_result.query_text.replace("Lidköping", "Linköping")
        if query_result.intent.end_interaction:
            self.end_of_dialog = True

        self.query_text_pub.publish(String(data=query_result.query_text))

        rospy.loginfo('-' * 10 + " %s " + '-' * 10, self.project_id)
        rospy.loginfo('Query text: {}'.format(query_result.query_text))
        rospy.loginfo('Detected intent: {} (confidence: {})\n'.format(
            query_result.intent.display_name,
            query_result.intent_detection_confidence))
        rospy.loginfo('Fulfillment text: {}\n'.format(
            query_result.fulfillment_text))

        
        if query_result.intent.display_name == "developer.linkopingMode":
            self.query_params = dialogflow.QueryParameters(geo_location = latlng_pb2.LatLng(latitude=58.4106611, longitude=15.6198244),
                                                           contexts = [dialogflow.Context(lifespan_count=100,
                                                                                         name="projects/"+self.project_id+"/agent/sessions/"+self.session_id+"/contexts/linkoping"
                                                           )]
            )
        elif query_result.intent.display_name == "developer.bergMode":
            self.query_params = dialogflow.QueryParameters(geo_location = latlng_pb2.LatLng(latitude=58.48548532662494, longitude=15.530466246782007),
                                                           contexts = [dialogflow.Context(lifespan_count=100,
                                                                                         name="projects/"+self.project_id+"/agent/sessions/"+self.session_id+"/contexts/berg"
                                                           )]
            )
        
        self.publish_response(query_result)

    def audio_stream_request_generator(self):
        """ Reads the sound from the ros-topic in an generator """
        query_input = dialogflow.QueryInput(audio_config=self.audio_config)
        # The first request contains the configuration.
        yield dialogflow.StreamingDetectIntentRequest(
            session=self.session,
            query_params=self.query_params,
            query_input=query_input)

        # Save data to audio file
        if self.save_audio_requests:
            filename = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+".wav"
            wf=wave.open(filename,"w")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
        # Here we are reading small chunks of audio from a queue
        while not rospy.is_shutdown() and not self.cancel_stream_intent:
            try:
                chunk = self.audio_chunk_queue.popleft()
            except IndexError as e:
                # Wait for new sound data, should come within 0.1s since it is sent in 10Hz
                rospy.sleep(0.1)
                continue
            if self.save_audio_requests:
                wf.writeframes(chunk)

            # The later requests contains audio data.
            yield dialogflow.StreamingDetectIntentRequest(input_audio=chunk)
        rospy.loginfo("AVBRÖT STREAMING INTENT!!")
        if self.save_audio_requests:
            wf.close()

    def playStartSound(self, hi=False):
        self.skip_audio = True
        if hi:
            if self.language == "sv":
                self.audio_play_srv(random.choice(["hejsan.wav","halla.wav","tja.wav"]),"")
            elif self.language == "en":
                self.audio_play_srv(random.choice(["hello.wav"]),"")
            elif self.language == "de":
                self.audio_play_srv(random.choice(["hallo.wav"]),"")
            elif self.language == 'zh-CN':
                self.audio_play_srv(random.choice(["nihao.wav"]),"")
        else:
            self.audio_play_srv("confirm_listen.wav","")
            time.sleep(0.1)
        self.skip_audio = False

    def playStopSound(self):
        self.skip_audio = True
        self.audio_play_srv("confirm_heard.wav","")
        self.skip_audio = False


    def run_until_sleep(self):
        isFirst = True
        while not rospy.is_shutdown():
            self.session_id = str(uuid.uuid4())
            self.session = self.session_client.session_path(self.project_id, self.session_id)
            rospy.loginfo('Session path: {}\n'.format(self.session))

            rospy.logwarn("VÄNTAR PÅ HOT WORD ELLER FACE!")
            start_waiting = time.time()
            while (not self.detected_wake_word and not self.head_visible) and not rospy.is_shutdown():
                rospy.sleep(0.1)
                if time.time() > start_waiting + 7:
                    rospy.logwarn("TIMEOUT, BACK TO SLEEP")
                    return
                if self.end_of_dialog:
                    rospy.logwarn("Got end of dialog, return 1")
                    return

            self.playStartSound(isFirst)
            isFirst = False
            self.is_listening_pub.publish(True)
            rospy.logwarn("SKICKAR LJUD TILL DIALOGFLOW")
            self.detect_intent_stream()
            self.is_listening_pub.publish(False)
            self.playStopSound()
            rospy.logwarn("VÄNTAR PÅ ATT ROBOTEN SKA PRATA KLART!")
            time.sleep(2) # Time to fetch website etc. So we don't skip this loop because we don't have gotten the is_talking flag imediately
            while self.is_talking and not rospy.is_shutdown():
                rospy.sleep(0.1)
            if self.end_of_dialog:
                rospy.logwarn("Got end of dialog, return 2")
                return
            rospy.sleep(0.7)

        
    def run(self):
        """ Update straming intents if we are using audio data """

        while not rospy.is_shutdown():
            # Create new session
            self.session_id = str(uuid.uuid4())
            self.session = self.session_client.session_path(self.project_id, self.session_id)
            rospy.loginfo('Session path: {}\n'.format(self.session))
            
            rospy.logwarn("VÄNTAR PÅ HOT WORD!")
            self.is_waiting_for_hot_word.publish(True)
            self.waiting_for_wake_word = True
            while not self.detected_wake_word and not rospy.is_shutdown():
                rospy.sleep(0.1)
            self.waiting_for_wake_word = False
            self.is_waiting_for_hot_word.publish(False)
            self.run_until_sleep()
            rospy.sleep(0.5)

if __name__ == '__main__':
    node = DialogflowNode()
    node.run()

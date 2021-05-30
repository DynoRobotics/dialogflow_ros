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
import wave
import time
from collections import deque
from google.cloud import dialogflow
from google.protobuf import struct_pb2
from google.api_core import exceptions

from std_msgs.msg import String, Bool, UInt16
from audio_common_msgs.msg import AudioData
from dialogflow_ros.msg import Response, Event, Context, Parameter

from std_srvs.srv import Empty, EmptyResponse

class DialogflowNode:
    """ The dialogflow node """
    def __init__(self):
        rospy.init_node('dialogflow_node')

        self.project_id = "qt-mega-agent-txtb"
        self.session_id = rospy.get_param('~session_id', uuid.uuid4())
        self.language = rospy.get_param('~default_language', 'sv-SE')
        self.disable_audio = rospy.get_param('~disable_audio', False)
        self.threshold = rospy.get_param('~threshold', 2000)
        self.time_before_start = rospy.get_param('~time_before_start', 0.5)
        self.save_audio_requests = rospy.get_param('~save_audio_requests', True)

        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(self.project_id, self.session_id)
        rospy.loginfo('Session path: {}\n'.format(self.session))

        self.audio_chunk_queue = deque(maxlen=int(self.time_before_start * 10)) # Times 10 since the data is sent in 10Hz

        # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
        audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
        sample_rate_hertz = 16000
        self.audio_config = dialogflow.InputAudioConfig(
            audio_encoding=audio_encoding,
            language_code=self.language,
            sample_rate_hertz=sample_rate_hertz,
            single_utterance=True)

        self.query_result_pub = rospy.Publisher('response', Response, queue_size=10)
        self.query_text_pub = rospy.Publisher('query_text', String, queue_size=10)
        self.transcript_pub = rospy.Publisher('transcript', String, queue_size=2)
        self.fulfillment_pub = rospy.Publisher('fulfillment_text', String, queue_size=10)
        self.listening_pub = rospy.Publisher('is_listening', Bool, queue_size=2)

        self.volume = 0
        self.is_talking = False
        self.stop_streaming = False
        rospy.Subscriber('text', String, self.text_callback)
        rospy.Subscriber('is_talking', Bool, self.is_talking_callback)
        rospy.Subscriber('event', Event, self.event_callback)

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
        contexts = contexts_client.list_contexts(self.session)
        for context in contexts:
            contexts_client.delete_context(context.name)
        return EmptyResponse()

    def is_talking_callback(self, msg):
        """ Callback for text input """
        self.is_talking = msg.data

    def text_callback(self, text_msg):
        """ Callback for text input """
        self.query_text_pub.publish(text_msg)
        query_result = self.detect_intent_text(text_msg.data)
        self.publish_response(query_result)

    def event_callback(self, event_msg):
        """ Callback for event input """
        rospy.loginfo("Publishing event %s", event_msg.name)
        query_result = self.detect_intent_event(event_msg)
        self.publish_response(query_result)

    def publish_response(self, query_result):
        """ Converts the dialogflow query result to the corresponding ros message """
        query_result_msg = Response()
        query_result_msg.project_id = self.project_id
        query_result_msg.query_text = query_result.query_text
        query_result_msg.intent_detection_confidence = query_result.intent_detection_confidence
        query_result_msg.intent.display_name = query_result.intent.display_name
        query_result_msg.intent.name = query_result.intent.name
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
        self.audio_chunk_queue.append(audio_chunk_msg.data)

    def volume_callback(self, msg):
        """ Callback for volume """
        self.volume = msg.data

    def detect_intent_text(self, text):
        """ Send text to dialogflow and publish response """
        self.stop_streaming = True
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
        self.stop_streaming = False
        return response.query_result

    def detect_intent_event(self, event_msg):
        """ Send event to dialogflow and publish response """
        self.stop_streaming = True
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
        self.stop_streaming = False
        return response.query_result

    def detect_intent_stream(self):
        """ Send streaming audio to dialogflow and publish response """
        if self.disable_audio:
            return

        requests = self.audio_stream_request_generator()
        self.listening_pub.publish(True)
        rospy.loginfo("STARTA LYSSNA!")
        responses = self.session_client.streaming_detect_intent(requests=requests)
        rospy.loginfo('=' * 10 + " %s " + '=' * 10, self.project_id)
        try:
            for response in responses:
                rospy.loginfo('Intermediate transcript: "{}".'.format(
                    response.recognition_result.transcript))
                self.transcript_pub.publish(response.recognition_result.transcript)
        except exceptions.OutOfRange as exc:
            rospy.logerr("Dialogflow exception. Out of audio quota? "
                         "No internet connection (%s)", exc)
            return
        self.listening_pub.publish(False)
        rospy.loginfo("SLUTA LYSSNA")

        # pylint: disable=undefined-loop-variable
        query_result = response.query_result

        if query_result.query_text == "":
            return

        self.query_text_pub.publish(String(data=query_result.query_text))

        rospy.loginfo('-' * 10 + " %s " + '-' * 10, self.project_id)
        rospy.loginfo('Query text: {}'.format(query_result.query_text))
        rospy.loginfo('Detected intent: {} (confidence: {})\n'.format(
            query_result.intent.display_name,
            query_result.intent_detection_confidence))
        rospy.loginfo('Fulfillment text: {}\n'.format(
            query_result.fulfillment_text))

        self.publish_response(query_result)

    def audio_stream_request_generator(self):
        """ Reads the sound from the ros-topic in an generator """
        query_input = dialogflow.QueryInput(audio_config=self.audio_config)
        # The first request contains the configuration.
        yield dialogflow.StreamingDetectIntentRequest(
            session=self.session,
            query_input=query_input)

        # Save data to audio file
        if self.save_audio_requests:
            filename = str(int(time.time()))+".wav"
            wf=wave.open(filename,"w")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
        # Here we are reading small chunks of audio from a queue
        while not rospy.is_shutdown() and not self.stop_streaming:
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

    def run(self):
        """ Update straming intents if we are using audio data """
        while not rospy.is_shutdown():
            if not self.is_talking:
                if self.volume > self.threshold:
                    self.detect_intent_stream()
            rospy.sleep(0.1)


if __name__ == '__main__':
    node = DialogflowNode()
    node.run()

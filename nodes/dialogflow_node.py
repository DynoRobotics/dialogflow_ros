#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# License: BSD
#   https://raw.githubusercontent.com/samiamlabs/dyno/master/LICENCE
#

import rospy
import uuid
import time
import queue
from google.cloud import dialogflow
from google.protobuf import struct_pb2

from std_msgs.msg import String, Bool
from qt_robot_interface.srv import speech_say
from audio_common_msgs.msg import AudioData
from dialogflow_ros.msg import Response, Intent, Event, Context, Parameter

from std_srvs.srv import Empty, EmptyResponse

class DialogflowNode:
    def __init__(self):
        rospy.init_node('dialogflow_node')

        self.project_id = "qt-mega-agent-txtb"
        self.session_id = rospy.get_param('~session_id', uuid.uuid4())
        self.language = rospy.get_param('~default_language', 'sv-SE')
        self.disable_audio = rospy.get_param('~disable_audio', False)

        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(self.project_id, self.session_id)
        rospy.loginfo('Session path: {}\n'.format(self.session))

        self.audio_chunk_queue = queue.Queue()
        #self.max_queue_size = 100

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
        self.record_pub = rospy.Publisher('is_recording', Bool, queue_size=1, latch=True)

        rospy.wait_for_service('qt_robot/speech/say')
        self.say_srv = rospy.ServiceProxy('qt_robot/speech/say', speech_say)

        rospy.Subscriber('/text', String, self.text_callback)
        rospy.Subscriber('event', Event, self.event_callback)

        if not self.disable_audio:
            rospy.Subscriber('/qt_robot/sound', AudioData, self.audio_callback)
            self.record_pub.publish(True)

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

    def handle_list_intents(self, request):
        intents_client = dialogflow.IntentsClient(credentials=self.credentials)
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

    def handle_list_context(self, request):
        contexts_client = dialogflow.ContextsClient(credentials=self.credentials)
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

    def handle_clear_context(self, request):
        contexts_client = dialogflow.ContextsClient(credentials=self.credentials)
        contexts = contexts_client.list_contexts(self.session)
        for context in contexts:
            contexts_client.delete_context(context.name)
        return EmptyResponse()


    def text_callback(self, text_msg):
        self.query_text_pub.publish(text_msg)
        query_result = self.detect_intent_text(text_msg.data)
        self.publish_response(query_result)

    def event_callback(self, event_msg):
        rospy.loginfo("Publishing event %s", event_msg.name)
        query_result = self.detect_intent_event(event_msg)
        self.publish_response(query_result)

    def publish_response(self, query_result):
        query_result_msg = Response()
        query_result_msg.project_id = self.project_id
        query_result_msg.query_text = query_result.query_text # text_msg.data
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

        for key in query_result.parameters:
            p_msg = Parameter()
            p_msg.key = key
            value = query_result.parameters.get(key)
            if isinstance(value, str):
                p_msg.value = [value]
            else:
                p_msg.value = value
            query_result_msg.parameters.append(p_msg)

        for ctx in query_result.output_contexts:
            name = ctx.name.split("/")[-1]
            if not name.startswith("__"):
                c_msg = Context()
                c_msg.name = name
                if ctx.parameters:
                    for key in ctx.parameters:
                        p_msg = Parameter()
                        p_msg.key = key
                        value = ctx.parameters.get(key)
                        if isinstance(value, str):
                            p_msg.value = [value]
                        else:
                            p_msg.value = value
                        c_msg.parameters.append(p_msg)
                query_result_msg.output_contexts.append(c_msg)

        self.query_result_pub.publish(query_result_msg)
        self.record_pub.publish(False)
        self.say_srv(query_result_msg.fulfillment_text)
        self.record_pub.publish(True)


    def audio_callback(self, audio_chunk_msg):
        #if len(self.audio_chunk_queue) < self.max_queue_size:

        self.audio_chunk_queue.put(audio_chunk_msg.data)
        #else:
        #    rospy.logwarn("Audio chunk queue if full, clearing!")
        #    self.audio_chunk_queue = Queue.Queue()

    def detect_intent_text(self, text):

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
        return response.query_result

    def detect_intent_event(self, event_msg):
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
        return response.query_result

    def detect_intent_stream(self):
        requests = self.audio_stream_request_generator()
        responses = self.session_client.streaming_detect_intent(requests=requests)
        rospy.loginfo('=' * 10 + " %s " + '=' * 10, self.project_id)
        try:
            for response in responses:
                rospy.loginfo('Intermediate transcript: "{}".'.format(
                    response.recognition_result.transcript))
        except:
            rospy.logerr("Dialogflow exception. Out of audio quota? "\
                         "No internet connection (project_id: %s)", self.project_id)
            return

        # Note: The result from the last response is the final transcript along
        # with the detected content.
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
        query_input = dialogflow.QueryInput(audio_config=self.audio_config)

        # The first request contains the configuration.
        yield dialogflow.StreamingDetectIntentRequest(
            session=self.session,
            query_input=query_input)

        # Here we are reading small chunks of audio from a dequeue
        while not rospy.is_shutdown():
            chunk = self.audio_chunk_queue.get()
            if not chunk:
                rospy.logwarn("##### Break loop")
                break
            # The later requests contains audio data.
            yield dialogflow.StreamingDetectIntentRequest(input_audio=chunk)

    def update(self):
        #if len(self.audio_chunk_queue) > 0:
        self.detect_intent_stream()


if __name__ == '__main__':
    dialogflow_node = DialogflowNode()
    while not rospy.is_shutdown():
        dialogflow_node.update()
        rospy.sleep(0.1)

cmake_minimum_required(VERSION 2.8.3)
project(dialogflow_ros)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  audio_common_msgs
  rospy
  roslint
  std_msgs
  std_srvs
)

add_message_files(
  DIRECTORY
    msg
  FILES
    Response.msg
    Event.msg
    Context.msg
    Parameter.msg
    Intent.msg
)

set(ROSLINT_PYTHON_OPTS
  "--max-line-length=127"
  "1>&2"
)
roslint_python()

generate_messages()

catkin_package()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

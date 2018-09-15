cmake_minimum_required(VERSION 2.8.3)
project(dialogflow_ros)

find_package(catkin REQUIRED message_generation)

add_message_files(
  DIRECTORY
    msg
  FILES
    QueryResult.msg
    Intent.msg
)

generate_messages()

catkin_package()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

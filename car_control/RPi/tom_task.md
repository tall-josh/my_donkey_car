# Wednesday task for Tom

- Make sure bluetooth controller is working again.
- Find where thumb sticks are being printed to the screen.
- Construct a Dictionary with the the following form:
  { "steering" : RIGHT_CONTOLLER_X_AXIS_VALUE,
    "throttle" : LEFT_CONTOLLER_Y_AXIS_VALUE
  }
- Add the code you have just made to the `tom_started_script.py` file.
- Use `json.dumps()` to turn the dict into a json string
- use `client.publish(TOPIC, JSON_STRING)` to sent a message with the topic, `"inference/control`
- Cross fingers
- Use this code block to write each steering command to a numbered json file. You will need to keep count each time the main loop runs to save each command under a different name, ie: frame_0001.json, frame_0002.json ...
  ```
  with open(SAVE_PATH, 'w') as my_file:
      json.dump(f, JSON_STRING)
  ``` 

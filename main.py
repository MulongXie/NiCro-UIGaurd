import os
import time
from datetime import datetime
import pathlib
import requests

from utils.RealDevice import RealDevice

#
# Get Physical Device action and previous UI page
#
# 1. run "adb.exe -d exec-out getevent -tl >> C:\Mulong\Code\Demo\Android\logs\actions.log"
# 2. run "python getAndroidActionPoint.py"
# 3. results saved to C:\android_dp\logs\output.log and C:\Mulong\Code\Demo\Android\screenshots


class AndroidAction:
    def __init__(self,
                 adb_path = r"adb.exe -d ",
                 root_save_path = r"C:\Mulong\Code\Demo\Android"):
        self.url = "http://localhost:80/sendAction"

        self.adb_path = adb_path
        self.root_save_path = root_save_path
        self.TMP_FOLDER = os.path.join(root_save_path, "screenshots")
        self.LOG_FOLDER = os.path.join(root_save_path, "logs")

        self.createFolder(self.TMP_FOLDER)
        self.createFolder(self.LOG_FOLDER)

        self.log_filename = os.path.join(self.LOG_FOLDER, "actions.log")
        # self.num_lines = len(open(self.log_filename, "r").readlines())

        self.watch_log_file = pathlib.Path(self.log_filename)
        self.previous_actions = []

        self.output_file = open(os.path.join(self.LOG_FOLDER, "output.log"), "w")
        # take screenshot and save in laptop
        self.takeScreenshot()

    def startMonitor(self):
        self.previous_time = None
        flag_take_screenshot = False
        while True:
            curr_time = self.watch_log_file.stat().st_mtime

            if self.previous_time != curr_time:
                self.previous_time = curr_time
                results = self.getAction()
                RealDevice(results)

    def createFolder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def takeScreenshot(self):
        current_dateTime = str(datetime.now()).replace(" ", "_").replace(":", "_")
        image_in_tmp = os.path.join(self.TMP_FOLDER, current_dateTime + ".png")
        os.system(self.adb_path + " exec-out screencap -p > " + str(image_in_tmp))
        self.previous_ui_path = image_in_tmp

    def getAction(self):
        with open(self.log_filename, "r", encoding = "utf-8") as f:
            all_logs = f.readlines()[::-1]
        # if len(all_logs) <= self.num_lines:
        #     return {}
        # tmp = len(all_logs)
        # all_logs = all_logs[self.num_lines:]

        points = []
        start_time, end_time, used_time = None, None, None
        flag_relase = False
        flag_new = True
        for line in all_logs:
            tokens = line.strip().replace("[", " ").replace("]", " ").split(" ")
            tokens = [tok for tok in tokens if tok != ""]

            # print("a", line.strip().replace("[", " ").replace("]", " "), "+++", tokens[0])
            if "BTN_TOOL_FINGER" in line and "UP" in line:
                # release action
                flag_relase = True
                points = []
                # print("up")
                end_time = float(tokens[0])
            if flag_relase:
                if "SYN_REPORT" in line:
                    flag_new = True
                elif "ABS_MT_POSITION_X" in line:
                    if flag_new:
                        points.append({"timestamp": tokens[0], "x": int(tokens[4], 16)})
                        flag_new = False
                    else:
                        points[-1]["x"] = int(tokens[4], 16)
                elif "ABS_MT_POSITION_Y" in line:
                    # print("y")
                    if flag_new:
                        points.append({"timestamp": tokens[0], "y": int(tokens[4], 16)})
                        flag_new = False
                    else:
                        points[-1]["y"] = int(tokens[4], 16)
                elif "DOWN" in line:
                    # print("down")
                    start_time = float(tokens[0])
                    used_time = end_time - start_time
                    break
            elif "DOWN" in line:
                print("action not finished")
                return {}

        if len(points) == 0:
            return {}

        # points = points[::-1]
        action_type = None
        first_point = []
        last_point = []
        if len(points) == 1:
            if used_time < 0.5:
                action_type = "Tap"
            else:
                action_type = "Long_Tap"
        else:
            first_x, first_y = None, None
            for point in points:
                if first_x is None and "x" in point:
                    first_x = point["x"]
                if first_y is None and "y" in point:
                    first_y = point["y"]
                if first_x and first_y:
                    break

            last_x, last_y = None, None
            for point in points[::-1]:
                if last_x is None and "x" in point:
                    last_x = point["x"]
                if last_y is None and "y" in point:
                    last_y = point["y"]
                if last_x and last_y:
                    break

            first_point = [first_x, first_y]
            last_point = [last_x, last_y]

            diff_horizontal =  last_x - first_x
            diff_vertical = last_y - first_y

            if abs(diff_horizontal) > abs(diff_vertical):
                if diff_horizontal > 0:
                    action_type = "Swipe_Right"
                else:
                    action_type = "Swipe_Left"
            else:
                if diff_vertical > 0:
                    action_type = "Swipe_Down"
                else:
                    action_type = "Swipe_Up"
        curr_action = {"points": points,
                       "action_type": action_type, "first_point": first_point,
                       "last_point": last_point}
        self.previous_actions.append(curr_action)

        curr_action["previous_UI_path"] = self.previous_ui_path

        if len(curr_action.get("points", [])) != 0:
            time.sleep(0.5)
            self.takeScreenshot()
            curr_action["curr_UI_path"] = self.previous_ui_path

        # self.sendActionData(curr_action)
        self.output_file.write(str(curr_action) + "\n")
        self.output_file.flush()
        return curr_action


if __name__ == "__main__":
    dump = AndroidAction()
    dump.startMonitor()

import json
import os
import cv2
from glob import  glob
from matplotlib import pyplot as plt
from datetime import  datetime
import numpy as np
cmap = plt.get_cmap("tab10")

# import jinja2
# import pdfkit

def draw_circle(number):
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img.fill(255)

    CENTER = (64, 64)

    cv2.circle(img, CENTER, 48, (0,0,255), -1)

    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1.5
    TEXT_THICKNESS = 7
    TEXT = str(number)

    text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)

    cv2.putText(img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)
    return img



def draw_results(img, output_results, img_path, output_folder, org_shape):
    h, w, _ = img.shape
    print(h, w, output_results)

    scale = h/org_shape[0]

    combined_img = np.zeros((h, 2*w, 3), dtype=np.uint8)
    combined_img.fill(255)
    combined_img[:h, :w] = img

    combined_img = cv2.putText(combined_img, str(datetime.now()), (w+5, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=3)

    thickness = 1
    TextScale = 0.8
    line_gap = 55
    circle_d = 30
    diff = 30
    num_text_per_line = 28
    if h < 1000:
        thickness = 1
        TextScale = 0.8
        line_gap = 40
        num_text_per_line = 20
        # diff = 30

    count = 1
    color_count = 0
    bbox2color = {}
    ins_id = 0
    x_offset = 0
    postfix = ""
    for DP_type, instances in output_results.items():
        if len(instances) == 0:
            continue
        postfix += "+{}@{}".format(DP_type, len(instances))
        for idx,item in enumerate(instances):
            curr_color = (0,0,255)#cmap(color_count % 10)[::-1]
            # curr_color = [int(255*a) for a in curr_color][:3]
            item["bbox"] = [int(a*scale) for a in item['container_bbox']]
            item['subType'] = item['type']
            x1,y1,x2,y2 = item["bbox"]
            subtype = item["subType"]
            if str(item["bbox"]) in bbox2color:
                curr_color = bbox2color[str(item["bbox"])]
                x_offset = circle_d
            else:
                bbox2color[str(item["bbox"])] = curr_color
            ins_id += 1

            if "children" in item:
                for child in item["children"]:
                    child_bbox = child["bbox"]
                    combined_img = cv2.rectangle(combined_img, (child_bbox[0],child_bbox[1]), (child_bbox[2],child_bbox[3]),
                                         color=curr_color, thickness=3)
            combined_img = cv2.rectangle(combined_img, (x1,y1), (x2,y2),
                                         color=curr_color, thickness=3)
            count += 1

            ## get circle
            circle = draw_circle(str(ins_id))
            circle = cv2.resize(circle, (circle_d,circle_d), interpolation = cv2.INTER_AREA)
            combined_img[max(0, y1-circle_d): max(0, y1-circle_d)+circle_d, max(0, x1-circle_d)+x_offset: max(0, x1-circle_d)+circle_d+x_offset] =  circle
            x_offset = 0

            combined_img[count*line_gap-diff: count*line_gap+(circle_d-diff), w+3:w+3+circle_d] = circle

            # subtype = subtype.split("]")[1]
            if y2-y1 > 0.7*h and x2-x1 > 0.7*w:
                subtype = "[Whole UI]" + subtype

            text_lines = len(subtype) // num_text_per_line + 1
            for text_line_id in range(text_lines):
                curr_text = subtype[text_line_id*num_text_per_line:(text_line_id+1)*num_text_per_line]
                # print(curr_text[-1], text_line_id, subtype[(text_line_id+1)*num_text_per_line])
                if curr_text[-1] != " " and text_line_id != text_lines-1 and subtype[(text_line_id+1)*num_text_per_line] != " ":
                    curr_text+="-"
                combined_img = cv2.putText(combined_img, curr_text, (w+30+40, 10+count*line_gap), cv2.FONT_HERSHEY_SIMPLEX, fontScale=TextScale, color=curr_color, thickness=thickness)
                count += 1
            color_count += 1
    if ins_id == 0:
        combined_img = cv2.putText(combined_img, "No detected malicious UI", (w+30+40, 10+count*line_gap), cv2.FONT_HERSHEY_SIMPLEX, fontScale=TextScale, color=(0,0,255), thickness=thickness)


    output_img_path = os.path.join(output_folder, os.path.basename(img_path).split(".")[0]+postfix+".png")
    cv2.imshow("darkpattern", combined_img)
    cv2.waitKey(0)
    cv2.destroyWindow('darkpattern')
    cv2.imwrite(output_img_path, combined_img)


if __name__ == "__main__":
    root_folder = r"vis_data"
    output_folder = root_folder.replace(r"\vis_data", "\output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_json = glob(root_folder+"/**.json")
    for json_file in all_json:
        img_path = json_file.split("_dp_")[0]+".png"

        img = cv2.imread(img_path)
        results = json.load(open(json_file, "r"))

        draw_results(img, results, img_path, output_folder)

    # render_pdf(output_folder)

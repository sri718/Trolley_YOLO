from ultralytics import YOLO

model = YOLO(r"best.pt")

new_class_names = {0: "Trolley"}
# since name was inappropriate in data.yaml while training

results = model("testimg.jpeg")

for result in results:
    result.names = new_class_names
    result.show() 
    # result.save() to store the output

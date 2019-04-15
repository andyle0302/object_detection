from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# parses the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-i", "--input", default=0,
	help="path to input video file")
ap.add_argument("-w", "--width", default=400,
	help="size of the max width")
args = vars(ap.parse_args())

# list of class labels model was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
IGNORE = set(["aeroplane"]) # list of ignored labels
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3)) # colors for bounding boxes

# loads our model from the disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream
print("[INFO] starting video stream...")

if args["input"] == "0":
	vs = VideoStream(src=0).start()
else:
	vs = FileVideoStream(path=args["input"]).start()

time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from video stream
	frame = vs.read()

	# resize, default max width is 400
	frame = imutils.resize(frame, width=int(args["width"]))

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# extract the index of the class labels
			idx = int(detections[0, 0, i, 1])

			if CLASSES[idx] in IGNORE:
				continue

			# compute the (x, y)-coordinates of the bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],	confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()
vs.stop()

import cv2

kitty_img = cv2.imread("../kitti_tiny/training/image_2/000028.jpeg")
cv2.rectangle(kitty_img, (147, 156), (205, 309), (0, 0, 255))
cv2.imshow('kitty', kitty_img)
cv2.waitKey()


kitty_img = cv2.imread("train/199.jpg")
cv2.rectangle(kitty_img, (510,156), (635,331), (0, 0, 255))
cv2.imshow('kitty', kitty_img)
cv2.waitKey()

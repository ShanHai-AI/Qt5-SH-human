# 表情识别
# 依赖于 https://github.com/oarriaga/paz
#  pip install pypaz -i  https://pypi.tuna.tsinghua.edu.cn/simple/

from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import paz.processors as pr

class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            box2D.class_name = self.classify(cropped_image)['class_name']
        return self.draw(image, boxes2D)


detect = EmotionDetector()
# you can now apply it to an image (numpy array)
def detect_emotion(image):
    predictions = detect(image)
    return predictions


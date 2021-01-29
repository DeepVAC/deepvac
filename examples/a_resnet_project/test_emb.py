from test import *
from deepvac import AugBase
from deepvac import LOG

class DeepvacNSFWEmb(DeepvacNSFW):
    def process(self):
        #Here the wa to let us just get 2048d feature before fc
        self.net.fc = nn.Sequential()
        # iterate all images
        for filename in self.dataset():
            #if 4 channel, to 3 channels
            self.sample = Image.open(filename).convert('RGB')
            # if use cv2 to read image
            #self.sample = AugBase.cv2pillow(self.sample)
            self.sample = self.conf.test.transform_op(self.sample).unsqueeze(0).to(self.conf.device)
            # forward
            self.output = self.net(self.sample)
            LOG.logI("feature shape: {}".format(self.output.shape))
            self.addEmb2DB(self.output)
            LOG.logI("feature db shape: {}".format(self.xb.shape))

        emb_file = "resnet50_gemfield_test.emb"
        LOG.logI("prepare to save db to {}".format(emb_file))
        self.saveDB(emb_file)
        LOG.logI("db file {} saved successfully.".format(emb_file))

if __name__ == '__main__':
    from config import config 
    nsfw_emb = DeepvacNSFWEmb(config)
    nsfw_emb()

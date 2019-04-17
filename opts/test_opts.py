from opts.base_opts import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--batch_size', default=1, type=int,help='mini-batch size (default: 256)')
        self.parser.add_argument('--crop_num', default=1, type=int,help='crop num')
        self.parser.add_argument('--epoch_num', default=0, type=int,help='which epoch load')

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--classind', default='dataset/ucf101_classInd.txt', type=str,help='classindex')

        self.isTrain = False
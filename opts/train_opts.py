from opts.base_opts import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--batch_size', default=16, type=int,help='mini-batch size (default: 256)')
        self.parser.add_argument('--larger_batch_size', default=16, type=int,help='backward several update once')
        self.parser.add_argument('--epochs', default=45, type=int,help='number of total epochs to run')
        self.parser.add_argument('--save_epoch_freq', default=1, type=int,help='number of total epochs to save')
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')

        self.parser.add_argument('--slow_rate', type=float, default=0.8, help='slow rate for weight update slower')
        self.parser.add_argument('--slow_lr_mult', type=float, default=0.1, help='slower update weight lr_mult')
        self.parser.add_argument('--slow_bn_mult', type=float, default=0.0, help='slower update weight lr_mult')
        self.parser.add_argument('--lr', default=0.001, type=float,help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
        self.parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,help='weight decay (default: 5e-4)')

        self.parser.add_argument('--lr_decay_epoch', type=int, default=20, help='iters change lr')
        self.parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='ratio change lr')


        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')


        self.isTrain = True
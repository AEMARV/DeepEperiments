import os
import theano
from Trainer import myTrainer
if __name__ == '__main__':
    os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu0,floatX=float32'
    print os.environ
    print theano.config.device
    myTrainer.run_trainer(cpu_debug=True)
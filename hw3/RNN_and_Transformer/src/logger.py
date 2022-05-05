import os
import logging
import sys
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)

def setup_default_logging(args, model_name= 'RNNLM',default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"):
    output_dir = os.path.join('logs',model_name)
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(comment=f'{model_name}-{args.learning_rate}')

    logger = logging.getLogger(model_name)

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, writer


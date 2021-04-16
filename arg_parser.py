import argparse

def training_parser():
    parser = argparse.ArgumentParser(description='Training arguments.')
    
    parser.add_argument('-lr', '--learning_rate', action='store',
                         default=0.01, type=float, help=('Learning Rate. Default: 0.01'))
    parser.add_argument('-bs', '--batch_size', action='store', 
                         default=200, type=int, help='Batch Size. Default: "200"')
    parser.add_argument('-ep', '--epochs', action='store', default=50, 
                         type=int, help=('Epochs. Default: 50'))
    return parser

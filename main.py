from arg_parser import training_parser
from training import main_loop

def main():
	args = training_parser().parse_args()

	LR                    = args.learning_rate
	EPOCHS                = args.epochs
	BATCH_SIZE            = args.batch_size
	main_loop(EPOCHS, BATCH_SIZE, LR)


if __name__ == '__main__':
    main()

import argparse
import yaml
from trainer import AutoEncoderTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select Model')
    parser.add_argument('--mode', '-M', type=str,
                        help='(draft, colorization, autoencoder)')
    args = parser.parse_args()

    with open('hyperparameters.yml') as yml:
        hp = yaml.load(yml, Loader=yaml.FullLoader)

    trainer = None

    if args.mode == 'draft':
        pass
    elif args.mode == 'colorization':
        pass
    elif args.mode == 'autoencoder':
        trainer = AutoEncoderTrainer(hp)
    else:
        raise NotImplementedError('mode : %s' % args.mode)

    trainer.train()

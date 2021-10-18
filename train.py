from vae_wrapper import load_vae_model
from vae.utils import parser
import os

args = parser()
trainer, vae_model, args, _ = load_vae_model(args)
trainer.fit(vae_model)

#check to make sure checkpoint directory exists otherwise create folder
if not os.path.exists('{}'.format(args.checkpoint_directory)):
    os.makedirs('{}'.format(args.checkpoint_directory))

trainer.save_checkpoint("{}{}_beta_{}.ckpt".format(args.checkpoint_directory, args.logging_string, args.beta))

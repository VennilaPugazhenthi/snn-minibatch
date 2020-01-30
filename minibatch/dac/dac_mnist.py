import os
import shutil
import argparse
from time import time
import tensorflow as tf #Changed

import torch
import numpy as np
from numpy import asarray,savetxt # Changed
import matplotlib.pyplot as plt
from torchvision import transforms  #transforms are for image transformations.
from torch.utils.tensorboard import SummaryWriter
from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015v2 #Changed here
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
from bindsnet.analysis.plotting import plot_spikes, plot_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
import ray.tune as tune


from minibatch.util import colorize, max_without_indices

accuracy=[] #Vennila: records the accuracy every 15 batches

#Vennila: returns the last accuracy value of training set
def end_accuracy():
    length= len(accuracy)
    return accuracy[length-1]

#Vennila: 
def tuning():
    tune.track.log()

def main(args):
    if args.update_steps is None:
        args.update_steps = max(250 // args.batch_size, 1)   #Its value is 16 # why is it always multiplied with step? #update_steps is how many batch to classify before updating the graphs

    update_interval = args.update_steps * args.batch_size   # Value is 240 #update_interval is how many pictures to classify before updating the graphs

    # Sets up GPU use
    torch.backends.cudnn.benchmark = False
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)   #to enable reproducability of the code to get the same result
    else:
        torch.manual_seed(args.seed)

    # Determines number of workers to use
    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    n_sqrt = int(np.ceil(np.sqrt(args.n_neurons)))

    if args.reduction == "sum":       #could have used switch to improve performance
        reduction = torch.sum           #weight updates for the batch
    elif args.reduction == "mean":
        reduction = torch.mean
    elif args.reduction == "max":
        reduction = max_without_indices
    else:
        raise NotImplementedError

    # Build network.
    network = DiehlAndCook2015v2(  #Changed here
        n_inpt=784,# input dimensions are 28x28=784
        n_neurons=args.n_neurons,
        inh=args.inh,
        dt=args.dt,
        norm=78.4,
        nu=(1e-4, 1e-2),
        reduction=reduction,
        theta_plus=args.theta_plus,
        inpt_shape=(1, 28, 28),
    )

    # Directs network to GPU
    if args.gpu:
        network.to("cuda")

    # Load MNIST data.
    dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=True,
        transform=transforms.Compose(   #Composes several transforms together
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )

    test_dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )

    # Neuron assignments and spike proportions.
    n_classes = 10 #changed
    assignments = -torch.ones(args.n_neurons) #assignments is set to -1
    proportions = torch.zeros(args.n_neurons, n_classes) #matrix of 100x10 filled with zeros
    rates = torch.zeros(args.n_neurons, n_classes)  #matrix of 100x10 filled with zeros

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.time) # Monitors:  Records state variables of interest. obj:An object to record state variables from during network simulation.
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)                     #state_vars: Iterable of strings indicating names of state variables to record.
                                                                                        #param time: If not ``None``, pre-allocate memory for state variable recording.
    weights_im = None
    spike_ims, spike_axes = None, None

    # Record spikes for length of update interval.
    spike_record = torch.zeros(update_interval, args.time, args.n_neurons)

    if os.path.isdir(args.log_dir):  #checks if the path is a existing directory
        shutil.rmtree(args.log_dir)     # is used to delete an entire directory tree

    # Summary writer.
    writer = SummaryWriter(log_dir=args.log_dir, flush_secs=60) #SummaryWriter: these utilities let you log PyTorch models and metrics into a directory for visualization
                                                                #flush_secs:  in seconds, to flush the pending events and summaries to disk.
    for epoch in range(args.n_epochs):  #default is 1
        print("\nEpoch: {epoch}\n")

        labels = []

        # Create a dataloader to iterate and batch data
        dataloader = DataLoader(    #It represents a Python iterable over a dataset
            dataset,
            batch_size=args.batch_size, #how many samples per batch to load
            shuffle=True,   #set to True to have the data reshuffled at every epoch
            num_workers=args.n_workers,
            pin_memory=args.gpu, #If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        )

        for step, batch in enumerate(dataloader): #Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object
            print("Step:",step)

            global_step = 60000 * epoch + args.batch_size * step

            if step % args.update_steps == 0 and step > 0:

                # Convert the array of labels into a tensor
                label_tensor = torch.tensor(labels)

                # Get network predictions.
                all_activity_pred = all_activity(
                    spikes=spike_record, assignments=assignments, n_labels=n_classes
                )
                proportion_pred = proportion_weighting(
                    spikes=spike_record,
                    assignments=assignments,
                    proportions=proportions,
                    n_labels=n_classes,
                )

                writer.add_scalar(
                    tag="accuracy/all vote",
                    scalar_value=torch.mean(
                        (label_tensor.long() == all_activity_pred).float()
                    ),
                    global_step=global_step,
                )
                #Vennila: Records the accuracies in each step
                value=torch.mean((label_tensor.long()==all_activity_pred).float())
                value=value.item()
                accuracy.append(value)
                print("ACCURACY:",value)
                writer.add_scalar(
                    tag="accuracy/proportion weighting",
                    scalar_value=torch.mean(
                        (label_tensor.long() == proportion_pred).float()
                    ),
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="spikes/mean",
                    scalar_value=torch.mean(torch.sum(spike_record, dim=1)),
                    global_step=global_step,
                )

                square_weights = get_square_weights(
                    network.connections["X", "Y"].w.view(784, args.n_neurons),
                    n_sqrt,
                    28,
                )
                img_tensor = colorize(square_weights, cmap="hot_r")

                writer.add_image(
                    tag="weights",
                    img_tensor=img_tensor,
                    global_step=global_step,
                    dataformats="HWC",
                )

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )

                labels = []

            labels.extend(batch["label"].tolist())  #for each batch or 16 pictures the labels of it is added to this list

            # Prep next input batch.
            inpts = {"X": batch["encoded_image"]}
            if args.gpu:
                inpts = {k: v.cuda() for k, v in inpts.items()} #.cuda() is used to set up and run CUDA operations in the selected GPU

            # Run the network on the input.
            t0 = time()
            network.run(inputs=inpts, time=args.time, one_step=args.one_step)   # Simulate network for given inputs and time.
            t1 = time() - t0

            # Add to spikes recording.
            s = spikes["Y"].get("s").permute((1, 0, 2))
            spike_record[
                (step * args.batch_size)
                % update_interval : (step * args.batch_size % update_interval)
                + s.size(0)
            ] = s

            writer.add_scalar(
                tag="time/simulation", scalar_value=t1, global_step=global_step
            )
            # if(step==1):
            #     input_exc_weights = network.connections["X", "Y"].w
            #     an_array = input_exc_weights.detach().cpu().clone().numpy()
            #     #print(np.shape(an_array))
            #     data = asarray(an_array)
            #     savetxt('data.csv',data)
            #     print("Beginning weights saved")
            # if(step==3749):
            #     input_exc_weights = network.connections["X", "Y"].w
            #     an_array = input_exc_weights.detach().cpu().clone().numpy()
            #     #print(np.shape(an_array))
            #     data2 = asarray(an_array)
            #     savetxt('data2.csv',data2)
            #     print("Ending weights saved")
            # Plot simulation data.
            if args.plot:
                input_exc_weights = network.connections["X", "Y"].w
               # print("Weights:",input_exc_weights)
                square_weights = get_square_weights(
                    input_exc_weights.view(784, args.n_neurons), n_sqrt, 28
                )
                spikes_ = {layer: spikes[layer].get("s")[:, 0] for layer in spikes}
                spike_ims, spike_axes = plot_spikes(
                    spikes_, ims=spike_ims, axes=spike_axes
                )
                weights_im = plot_weights(square_weights, im=weights_im)

                plt.pause(1e-8)

            # Reset state variables.
            network.reset_state_variables()
        print(end_accuracy()) #Vennila

    #To calculate the Testing accuracy
    # for epoch in range(args.n_epochs):  #default is 1
    #     print("\n Testing\n")
    #
    #     labels = []
    #
    #     # Create a dataloader to iterate and batch data
    #     dataloader = DataLoader(    #It represents a Python iterable over a dataset
    #         test_dataset,
    #         batch_size=args.batch_size, #how many samples per batch to load
    #         shuffle=True,   #set to True to have the data reshuffled at every epoch
    #         num_workers=args.n_workers,
    #         pin_memory=args.gpu, #If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    #     )
    #
    #     for step, batch in enumerate(dataloader): #Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object
    #         print("Step:",step)
    #
    #         global_step = 10000 * epoch + args.batch_size * step
    #
    #         if step % args.update_steps == 0 and step > 0:
    #             # Convert the array of labels into a tensor
    #             label_tensor = torch.tensor(labels)
    #
    #             # Get network predictions.
    #             all_activity_pred = all_activity(
    #                 spikes=spike_record, assignments=assignments, n_labels=n_classes
    #             )
    #             proportion_pred = proportion_weighting(
    #                 spikes=spike_record,
    #                 assignments=assignments,
    #                 proportions=proportions,
    #                 n_labels=n_classes,
    #             )
    #
    #             writer.add_scalar(
    #                 tag="testing accuracy/all vote",
    #                 scalar_value=torch.mean(
    #                     (label_tensor.long() == all_activity_pred).float()
    #                 ),
    #                 global_step=global_step,
    #             )
    #             writer.add_scalar(
    #                 tag="testing accuracy/proportion weighting",
    #                 scalar_value=torch.mean(
    #                     (label_tensor.long() == proportion_pred).float()
    #                 ),
    #                 global_step=global_step,
    #             )
    #             writer.add_scalar(
    #                 tag="testing spikes/mean",
    #                 scalar_value=torch.mean(torch.sum(spike_record, dim=1)),
    #                 global_step=global_step,
    #             )
    #
    #             square_weights = get_square_weights(
    #                 network.connections["X", "Y"].w.view(784, args.n_neurons),
    #                 n_sqrt,
    #                 28,
    #             )
    #             img_tensor = colorize(square_weights, cmap="hot_r")
    #
    #             writer.add_image(
    #                 tag="weights",
    #                 img_tensor=img_tensor,
    #                 global_step=global_step,
    #                 dataformats="HWC",
    #             )
    #
    #             # Assign labels to excitatory layer neurons.
    #             assignments, proportions, rates = assign_labels(
    #                 spikes=spike_record,
    #                 labels=label_tensor,
    #                 n_labels=n_classes,
    #                 rates=rates,
    #             )
    #
    #             labels = []
    #
    #         labels.extend(batch["label"].tolist())  #for each batch or 16 pictures the labels of it is added to this list
    #
    #         # Prep next input batch.
    #         inpts = {"X": batch["encoded_image"]}
    #         if args.gpu:
    #             inpts = {k: v.cuda() for k, v in inpts.items()} #.cuda() is used to set up and run CUDA operations in the selected GPU
    #
    #         # Run the network on the input.
    #         t0 = time()
    #         network.run(inputs=inpts, time=args.time, one_step=args.one_step)   # Simulate network for given inputs and time.
    #         t1 = time() - t0
    #
    #         # Add to spikes recording.
    #         s = spikes["Y"].get("s").permute((1, 0, 2))
    #         spike_record[
    #             (step * args.batch_size)
    #             % update_interval : (step * args.batch_size % update_interval)
    #             + s.size(0)
    #         ] = s
    #
    #         writer.add_scalar(
    #             tag="time/simulation", scalar_value=t1, global_step=global_step
    #         )
    #         # if(step==1):
    #         #     input_exc_weights = network.connections["X", "Y"].w
    #         #     an_array = input_exc_weights.detach().cpu().clone().numpy()
    #         #     #print(np.shape(an_array))
    #         #     data = asarray(an_array)
    #         #     savetxt('data.csv',data)
    #         #     print("Beginning weights saved")
    #         # if(step==3749):
    #         #     input_exc_weights = network.connections["X", "Y"].w
    #         #     an_array = input_exc_weights.detach().cpu().clone().numpy()
    #         #     #print(np.shape(an_array))
    #         #     data2 = asarray(an_array)
    #         #     savetxt('data2.csv',data2)
    #         #     print("Ending weights saved")
    #         # Plot simulation data.
    #         if args.plot:
    #             input_exc_weights = network.connections["X", "Y"].w
    #            # print("Weights:",input_exc_weights)
    #             square_weights = get_square_weights(
    #                 input_exc_weights.view(784, args.n_neurons), n_sqrt, 28
    #             )
    #             spikes_ = {layer: spikes[layer].get("s")[:, 0] for layer in spikes}
    #             spike_ims, spike_axes = plot_spikes(
    #                 spikes_, ims=spike_ims, axes=spike_axes
    #             )
    #             weights_im = plot_weights(square_weights, im=weights_im)
    #
    #             plt.pause(1e-8)
    #
    #         # Reset state variables.
    #         network.reset_state_variables()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-neurons", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--update-steps", type=int, default=None)
    parser.add_argument("--inh", type=float, default=120)
    parser.add_argument("--theta_plus", type=float, default=0.05)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--dt", type=int, default=1.0)
    parser.add_argument("--intensity", type=float, default=128)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--one-step", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

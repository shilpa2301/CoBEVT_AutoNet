import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys


# Open the file in read-binary mode
with open('bev_all.pkl', 'rb') as file:
    # Load the variable from the file
    bev_all = pickle.load(file)

print("BEV loaded successfully!")

num_cav = bev_all.shape[0]
# print(bev_all[1])
# sys.exit()


for i in range(num_cav):
    print("Processing vehicle", i)
    bev_data = bev_all[i].cpu().detach().numpy()

    # Select a feature channel to visualize
    # channel_to_visualize = 0
    for channel_to_visualize in range(128):
        bev_channel = bev_data[channel_to_visualize]

        # Normalize the data for visualization
        bev_channel_normalized = (bev_channel - bev_channel.min()) / (bev_channel.max() - bev_channel.min())

        # Plot using matplotlib
        plt.imshow(bev_channel_normalized, cmap='viridis')
        plt.title(f'BEV Channel {channel_to_visualize}')
        plt.colorbar()

        # Save the image to a file
        plt.savefig('bev_'+str(i)+'/'+str(channel_to_visualize)+'.png')

        # Clear the current figure
        plt.clf()

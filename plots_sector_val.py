import pandas as pd
import matplotlib.pyplot as plt
import configs
# Load the CSV file into a DataFrame
df = pd.read_csv('corrected_mean_velocity_bowl_{}_{}_secs.csv'.format(configs.bowl_name, configs.average_per_seconds))
# ax = plt.figure(figsize=(5, 6))
fig, ax = plt.subplots()
# Extract region names and velocity data
regions = df['sectors']
velocities = df.iloc[:, 1:]  # Exclude the first column (Sectors) for plotting
print("velocities are :", velocities)
# Create a line plot for each region
for i in range(len(regions)):
    plt.plot(velocities.iloc[i], label=regions.iloc[i])

# Customize the plot
plt.title('Average Velocities of parts in Bowl {} over {} seconds (for {} seconds each)'.format(configs.bowl_name, configs.actual_video_length, configs.average_per_seconds))
plt.xlabel('Seconds')
plt.ylabel('Average Velocity (pixels/second)')
plt.xticks(range(configs.actual_video_length), ['{}'.format(h) for h in range(0, configs.actual_video_length+configs.average_per_seconds, configs.average_per_seconds)])
# plt.xticks([0, 10, 20, 30, 40, 50], ['{}'.format(h) for h in range(0, configs.actual_video_length+configs.average_per_seconds, configs.average_per_seconds)] )
ax.autoscale(enable=None, axis="x", tight=True)

plt.legend()

# Show the plot
# plt.grid(True)
# plt.tight_layout()
plt.show()
# plt.savefig("plot_bowl_{}_{}_secs.png".format(configs.bowl_name, configs.average_per_seconds))

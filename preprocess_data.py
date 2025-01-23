# %%
import climnet.community_detection.graph_tool.gt_functions as gtf
import climnet.community_detection.graph_tool.es_graph_tool as egt
import geoutils.utils.file_utils as fut
import geoutils.tsa.event_synchronization as es
import numpy as np
import xarray as xr
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.plotting.plots as gplt
import geoutils.preprocessing.open_nc_file as of
import climnet.network.clim_networkx as nx
import climnet.datasets.evs_dataset as evs

from importlib import reload


plot_dir = "/home/strnad/data/plots/extremes/"
data_dir = "/home/strnad/data/"
# %%
# Load the data
pr_file_path = f"{data_dir}/climate_data/1/mswep_pr_{1}_ds.nc"
ds_pr = of.open_nc_file(pr_file_path)
variable = 'pr'
da_pr = ds_pr[variable]
start_date, end_date = tu.get_start_end_date(da_pr)
# %%
# Restrict to Indian subcontinent
lon_range = [65, 95]
lat_range = [5, 35]

da_india = sput.cut_map(da_pr, lon_range, lat_range)

mean_pr = da_india.mean(dim=["time"])
im = gplt.plot_map(mean_pr, title="Mean precipitation",
                   label="precipitation [mm/day]")
# %%
dates = tu.get_dates_of_time_range(time_range=[start_date,
                                   end_date], freq="D")

start_month = 'Jun'
end_month = 'Sep'
ds_india_month_range = tu.get_month_range_data(
    da_india, start_month, end_month,
    set_zero=True,)
ds_india_jjas = tu.get_month_range_data(
    da_india, start_month, end_month,
    set_zero=False,)

dates_in_range = tu.get_month_range_data(dates, start_month, end_month)


# %%
# Create event series dataset by threshold 0.9 and an ERE should have at least 10 mm/day and at least per cell should be minimum 10 events over the full period
reload(tu)
reload(gplt)
q = 0.9
th_eev = 15
min_num_events = 10

evs_india, mask = tu.compute_evs(dataarray=ds_india_month_range,
                                 q=q,
                                 min_num_events=min_num_events,
                                 th_eev=th_eev,
                                 min_threshold=1  # This is important for the EREs
                                 )
q_map = tu.get_q_val_map(dataarray=da_india, q=q)

num_eres = evs_india.sum(dim='time')


# %%
# Plot the data to get some idea about the number of EREs
fdic = gplt.create_multi_plot(
    1, 2,
    figsize=(8, 8),
    wspace=0.25,
    projection="PlateCarree")
gplt.plot_map(
    q_map,
    ax=fdic["ax"][0],
    significance_mask=xr.where(mask, 0, 1),
    plot_type="colormesh",
    label=rf"$Q_{{pr}}({{{q}}})$ precipitation [mm/day]",
    vmin=0, vmax=25, levels=20, tick_step=4,
    cmap="RdYlBu",
)

_ = gplt.plot_map(
    num_eres,
    ax=fdic["ax"][1],
    significance_mask=xr.where(mask, 0, 1),
    plot_type="colormesh",
    label=rf"Number of EREs",
    vmin=100, vmax=500, levels=20, tick_step=4,
    cmap="viridis",
)

# %%
# Create the network of EREs
reload(es)

# First create the null model (is only needed once for a particular length of the time points)
# Attention events per month are not uniform distributed
length = len(dates_in_range)
num_permutations = 1000
num_eres_month_range = evs_india.sum(dim='time')
max_num_events = int(np.max(num_eres_month_range))

savepath_null_model = f'null_models/null_model_{length}_{
    max_num_events}_{num_permutations}.npy'
# %%
reload(es)
if fut.exist_file(savepath_null_model):
    null_model_dict = fut.load_np_dict(savepath_null_model)
else:
    null_model_dict = es.null_model_distribution(
        length_time_series=length,
        max_num_events=max_num_events*0.8,
        num_permutations=num_permutations,
        savepath=savepath_null_model,
    )
    fut.save_np_dict(null_model_dict,
                     savepath_null_model)
# %%
# Plot the null model
reload(gplt)
im = gplt.plot_array(z=null_model_dict[0.95],
                     label='Number of sync. events',
                     xlabel='Number of events i',
                     ylabel='Number of events j',
                     orientation='vertical',)


# %%
# First prepare the data to a dataset with numbers per cell to be better traceable in the network setting
reload(evs)
grid_type = 'fekete'
grid_step = 1
sp_grid = f'{grid_type}_{grid_step}.npy'
ds_evs = evs.EvsDataset(
    data=evs_india,
    grid_type=grid_type,
    sp_grid=sp_grid,
)
taumax = 10
ds = ds_evs
# %%
# Init the network
reload(nx)
Net = nx.Clim_NetworkX(dataset=ds,
                       taumax=taumax,
                       )

# Run the event synchronization algorithm to all points of the network
Net.create(
    method='es',
    null_model_file=savepath_null_model,
    E_matrix_folder='./E_matrix/',
    q_sig=0.95,  # significance level of null model
)

# %%
# plot links of the network
# Plot the network links for a certain location
reload(gplt)
lat_range = [23, 25]
lon_range = [75, 80]
link_dict = Net.get_edges_nodes_for_region(
    lon_range=lon_range, lat_range=lat_range, binary=False
)

# Plot nodes where edges go to
im = gplt.plot_map(
    link_dict['target_map'],
    ds=Net.ds,
    label=f"Local degree",
    projection="PlateCarree",
    plot_type="colormesh",
    cmap="Greens",
    vmin=0,
    vmax=10,
    levels=10,
    alpha=0.7,
)

im = gplt.plot_edges(
    Net.ds,
    link_dict['el'][::5],  # plot every fifth edge
    ax=im["ax"],
    lw=0.2,
    alpha=0.6,
)

gplt.plot_rectangle(
    ax=im["ax"],
    lon_range=lon_range,
    lat_range=lat_range,
    color="magenta",
    lw=3,
    zorder=11
)
savepath = f"{plot_dir}/network_plots/india_{lon_range}_links.png"
gplt.save_fig(savepath)
# %%
# Apply community detection to the network
reload(gtf)
reload(egt)
num_communities = 5
res_dict = gtf.apply_SBM(Net.cnx, B_max=num_communities,
                         multi_level=False)

# %%
# plot the communities
reload(gplt)
hard_cluster = res_dict['group_levels'][0]  # theta['node_levels'][0]
hc_map = Net.ds.get_map(hard_cluster)
im = gplt.plot_map(
    hc_map,
    ds=Net.ds,
    vmin=0, vmax=num_communities,
    levels=num_communities,
    title=f'SBM for {num_communities} communities',
    significance_mask=True,
    plot_type="colormesh",
    projection="PlateCarree",
    cmap="rainbow",
    label="Group number",
    tick_step=1,
)

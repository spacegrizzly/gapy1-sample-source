"""
3D FoV (stacked) and Reflected Analysis
===========

This is a standard script for analysing gamma-ray sources using python,
and in particular the gammapy=1.0/1.1 package

The foundations of this script originated from the 3D analysis script
documented here:
https://docs.gammapy.org/1.0/tutorials/analysis-3d/analysis_3d.html


Author: Christopher Burger-Scheidlin
Email: cburger@cp.dias.ie
Date: 28 May 2024


Input:
    * config.toml that includes information on
        * Path to datastore
        * Path to high galactic plane survey FITS file (optional)
        * a dataset of the source (optional, can be created in script and read)
        * a runlist of the target source (optional, can be created in script)

Output:
    * a subdirectory containing:
        - the (stacked) dataset(s)
        - plots generated with this script
        - FITS files of the analysis, including fit results as tables
        - a runlist.txt file of the runs used
        - list of background normalisation information and problematic runs
          as *.txt files
        - the configuration of the analysis as *.log file
        - a csv file containing fit parameters from modelling
        - a log file of the terminal output (optional)

"""

######################################################################
# Import packages
# -----------

import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import table
from astropy.visualization.wcsaxes import SphericalCircle
from regions import CircleSkyRegion

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import gammapy
from gammapy.catalog import SourceCatalogHGPS
from gammapy.data import DataStore
from gammapy.datasets import MapDataset, Datasets, SpectrumDataset
from gammapy.estimators import ExcessMapEstimator, FluxPointsEstimator
from gammapy.makers import (
    MapDatasetMaker,
    SpectrumDatasetMaker,
    SafeMaskMaker,
    FoVBackgroundMaker,
    ReflectedRegionsBackgroundMaker,
    RingBackgroundMaker
)
from gammapy.maps import MapAxis, WcsGeom, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    SkyModel,
    PointSpatialModel,
    DiskSpatialModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    LogParabolaSpectralModel,
    FoVBackgroundModel,
)
from gammapy.visualization import plot_npred_signal

import argparse
from itertools import compress
import pandas as pd
import pickle
from pprint import pprint
from time import time
import tomllib
import warnings


######################################################################
# Setup to have coloured output
# -----------

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_green(*args: str):
    print(bcolors.OKGREEN, *args, bcolors.ENDC)


def print_yellow(*args: str):
    print(bcolors.WARNING, *args, bcolors.ENDC)


def print_red(*args: str):
    print(bcolors.FAIL, *args, bcolors.ENDC)


def pandas_wide():
    # import pandas as pd
    desired_width = 200
    pd.set_option('display.width', desired_width)
    # For series
    pd.set_option('display.max_seq_items', None)
    # For DataFrame columns
    pd.set_option('display.max_columns', None)


def save_or_show_plot(filename: str) -> None:
    """
    This function plots whatever is currently present in plt (matplotlib.plot).

    :param filename: str, name of the file to export as *.png or plot
    :return: None
    """
    if export_plots:
        plt.savefig(path / f"{filename}_{std_filename}.png")
    if show_plots:
        plt.show()
    plt.close()

    return None


def load_toml(file: str) -> dict:
    """Load TOML data from file"""
    with open(f"{file}", "rb") as f:
        toml_data: dict = tomllib.load(f)
        return toml_data


######################################################################
# Time the script
# -----------

start_total = time()

######################################################################
# Check gammapy version
# -----------

print_green("gammapy " + gammapy.__version__)
if gammapy.__version__ == '1.0' or gammapy.__version__ == '1.1':
    print("Using accepted gammapy version.")
else:
    raise Exception("Gammapy version 1.0/1.1 required")

######################################################################
# Read in user configuration with the option of passing a file as an
# argument
# -----------

parser = argparse.ArgumentParser(
    description="Add a toml file to specify the filename of the configuration."
)
parser.add_argument("-f", "--file",
                    required=False,
                    type=str,
                    help="Specify the toml filename for the configuration "
                         "(optional). Default is 'config.toml'.")
args = parser.parse_args()
filename = args.file

# Test if an argument was passed for the config file. If not the default toml
# will be used. If yes, then the specified file will be set as config file.
if filename is None:
    print_yellow("No config file specified. Using default 'config.toml'.")
    config = load_toml("config.toml")
else:
    print_green(f"Config file found. Loading '{filename}'.")
    config = load_toml(filename)

pandas_wide()

# Assign variables
export_plots: bool = config["plot"]["export"]
show_plots: bool = config["plot"]["show"]
dpi: int = config["plot"]["dpi"]

######################################################################
# Change global matplotlib settings
# -----------

font = {
    # 'family': 'normal',
    # 'weight': 'normal',
    'size': 10
}

figure = {"figsize": (8, 6)}

matplotlib.rc('font', **font)
matplotlib.rc('figure', **figure)

######################################################################
# Input Source Information
# -----------

# Read from user_config
source_name: str = config["source"]["name"]
radius: float = config["source"]["radius"]
bkg_method: str = config['background']["method"]
analysis_version: str = config["main"]["analysis_version"]

# Create std_filename
std_filename = f"{source_name}_{radius}_{bkg_method}_v{analysis_version}"

# Create custom subfolder to save the analysis results
path = Path(f"analysis_{std_filename}")
path.mkdir(exist_ok=True)

if config["main"]["create_logfile"]:
    # Start logging console output
    # sys.stdout = open(path / f"screenlog_{std_filename}.log", "w")

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(path / f"screenlog_{std_filename}.log", "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            # you might want to specify some extra behavior here.
            # self.log.close()
            pass


    sys.stdout = Logger()
    # # sys.stderr = sys.stdout

pprint(config, sort_dicts=False)
print(f"Path: {path}\n")

print_green("Set source position ...")
if config["source"]["get_position_from_config"]:
    print_green("\t\t ... from user_config")
    lon = config["source"]["lon"]
    lat = config["source"]["lat"]
    frame = config["source"]["frame"]

    position = SkyCoord(lon, lat, frame=frame, unit="deg")
    position = position.galactic

else:
    print_green("\t\t ... from source name")
    position = SkyCoord.from_name(source_name)
    position = position.galactic

# Finalise the position and coordinates to use in remainder of script
lon = position.l.deg
lat = position.b.deg
# Frame has to be galactic, as all positions are converted in the previous step
frame = "galactic"
radius = config["source"]["radius"]

# Save the position to the config, which will later be saved to disk as well
config["source"]["lon"] = lon
config["source"]["lat"] = lat

# Create an observation cone around the position
cone_radius = config["runlist"]["cone_radius"]
observation_cone = dict(
    type="sky_circle",
    frame="galactic",
    lon=f"{lon} deg",
    lat=f"{lat} deg",
    radius=f"{cone_radius} deg",
)
print_yellow(f"{source_name} \t {position}\n")

######################################################################
# Check setup
# -----------

if config["main"]["check_tutorials_setup"]:
    print_green("Checking tutorials setup ...")
    from gammapy.utils.check import check_tutorials_setup

    check_tutorials_setup()

######################################################################
# Obtain and analyse runlist
# -----------

print_green("Obtaining runlist ... ")

# Load the required datastore
if config["datastore"]["filename"] == "nan":
    datastore = DataStore.from_dir(Path(config["datastore"]["path"]))

else:
    try:
        datastore = DataStore.from_dir(
            base_dir=Path(config["datastore"]["path"]),
            hdu_table_filename=config["datastore"]["filename"])
        # todo hdu table problem?
    except OSError:
        datastore_filename = config["datastore"]["filename"]
        print_red("Name of datastore file (hdu_table_filename) is invalid.")
        print_red("Should be either the name of the file, or str(nan).")
        print_red(f"It is, however, {datastore_filename}")
        print_red("If filename is correct, check path to datastore.")
        sys.exit(1)

if config["runlist"]["create_in_file"]:
    print_green("Creating runlist in file from observation cone ... \n")

    # Get runlist from the observation cone and put cuts on the runs
    selected_obs = datastore.obs_table.select_observations(observation_cone).to_pandas()

else:
    print_green("Loading runlist from disk ... ")

    # If no special name is specified for the filename of the runlist load the
    # default runlist name. If a filename is specified, load that one from
    # within the analyis* folder

    if config["runlist"]["filename"] == "nan":
        print_green("\t\t ... using standard name")
        loaded_runlist = list(np.loadtxt(path / f"runlist_{std_filename}.txt", dtype=int))
    else:
        runlist_filename = config["runlist"]["filename"]
        print_green(f"\t\t ... using custom name: {runlist_filename}")
        loaded_runlist = list(np.loadtxt(path / runlist_filename, dtype=int))

    try:
        selected_obs = datastore.obs_table.select_obs_id(loaded_runlist).to_pandas()
    except AttributeError:
        print("AttributeError raised when trying to convert loaded runlist to pd.Dataframe.")
        selected_obs = datastore.obs_table.select_obs_id(loaded_runlist)
        selected_obs = pd.DataFrame(np.array(selected_obs))

if config["runlist"]["filter"]["run"]:
    # Apply runlist filters

    if config["runlist"]["filter"]["max_zenith"] != "nan":
        # Apply runlist filter: Zenith angle limitation
        max_zenith = config["runlist"]["filter"]["max_zenith"]
        print_green(f"Applying filter for max. zenith angle (deg):\t{max_zenith}")

        selected_obs = selected_obs[
            selected_obs["ZEN_PNT"] <= max_zenith]
    else:
        print_yellow("No runlist max_zenith filter applied ...")

    if config["runlist"]["filter"]["years"] != ["nan"]:
        # Apply runlist filter: select years from which the raw data stems

        # Parse dates in obs table
        selected_obs["date"] = selected_obs["DATE-OBS"].str.decode("ascii")
        selected_obs["date"] = pd.to_datetime(selected_obs["date"])

        years = config["runlist"]["filter"]["years"]
        print_green(f"Applying filter for years:\t\t\t{years}")

        if len(years) == 1:
            selected_obs = selected_obs[
                selected_obs["date"].between(f"{years[0]}-01-01", f"{years[0]}-12-31")]
        elif len(years) == 2:
            selected_obs_table_1 = selected_obs[
                selected_obs["date"].between(f"{years[0]}-01-01", f"{years[0]}-12-31")]
            selected_obs_table_2 = selected_obs[
                selected_obs["date"].between(f"{years[1]}-01-01", f"{years[1]}-12-31")]

            selected_obs = pd.concat(
                [selected_obs_table_1, selected_obs_table_2], ignore_index=True)
            del selected_obs_table_1, selected_obs_table_2


        else:
            raise Exception("No more than 2 params for runlist option 'filter.year' supported")

    else:
        print_yellow("No runlist yearly filter applied ...")

    if config["runlist"]["filter"]["duration"] != "nan":
        # Apply runlist filter: Filter out runs shorter (!) than the value given
        duration = config["runlist"]["filter"]["duration"]
        print_green(f"Applying filter for duration of observations:\t > {duration}")
        selected_obs = selected_obs[selected_obs["LIVETIME"] >= duration]
    else:
        print_yellow("No runlist duration filter applied ...")

    if config["runlist"]["filter"]["number"] != "nan":
        # Apply runlist filter: Limit the number of runs beginning from the lowest number
        number = config["runlist"]["filter"]["number"]
        print_green(f"Applying filter for number of observations:\t{number}")
        selected_obs = selected_obs[:number]
    else:
        print_yellow("No runlist numbers filter applied ...")

    # Space for some restrictions/filtering on the run list using pandas i.e.

# Runlist ist finalised. Print runlist here.
runlist = list(selected_obs.OBS_ID)

# Add the selected observation IDs to the user config dictionary
config["runlist"]["runlist"] = runlist
counter_observations = len(selected_obs)

# Save runlist to text file
pd.DataFrame(runlist).to_csv(
    path / f"runlist_{std_filename}.txt",
    sep="\n", index=False, header=False)

# Print information on selected runlist
print_yellow("Length\t" + str(len(runlist))
             + "\t\t" + str(runlist))

######################################
# Parsing colours for plotting
# -----------

colour_list = config["plot"]["colours"]

######################################
# Plot zenith angle distribution
# -----------

plt.figure()
plt.hist(selected_obs["ZEN_PNT"],
         density=False, bins=40, color=colour_list[2], ec="xkcd:black")
plt.xlabel('Zenith angle (deg)')
plt.ylabel('Number of observations')
plt.text(x=selected_obs["ZEN_PNT"].max(), y=2,
         s=f"Total runs:  {len(selected_obs)}", horizontalalignment='right')

save_or_show_plot("zenith-angle")

# Print some time info
print("ONTIME:\t\t\t", round(selected_obs.ONTIME.sum() / 3600, 2), "h",
      "\t\t", round(selected_obs.ONTIME.sum() / (3600 * 24), 2), "days")

######################################################################
# Preparing reduced datasets geometry
# -----------

energy_axis = MapAxis.from_energy_bounds(
    config["energy"]["low"],
    config["energy"]["high"],
    config["energy"]["nbin"],
    unit="TeV")

# Reduced IRFs are defined in true energy (i.e. not measured energy).
energy_axis_true = MapAxis.from_energy_bounds(
    config["energy"]["true_low"],
    config["energy"]["true_high"],
    config["energy"]["true_nbin"],
    unit="TeV", name="energy_true")

# Make sure that interval of `energy true` fully includes `energy` !
assert config["energy"]["true_low"] < config["energy"]["low"]
assert config["energy"]["high"] < config["energy"]["true_high"]

print("energy_axis", energy_axis)
print("energy_axis_true", energy_axis_true)

if config["background"]["method"] == "fov":
    wcsgeom = WcsGeom.create(
        skydir=position,
        binsz=config["map"]["size_binsz"],
        width=(config["map"]["size"], config["map"]["size"]),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis],  # this line is essential for FoV background method
    )

    # Now we can define the target dataset with this geometry.
    dataset_stacked = MapDataset.create(geom=wcsgeom,
                                        energy_axis_true=energy_axis_true,
                                        name=f"{source_name}-stacked")


elif config["background"]["method"] == "reflected":

    geom = RegionGeom.create(
        region=f"galactic;circle({lon},{lat},{radius})",
        axes=[energy_axis])

    wcsgeom = WcsGeom.create(
        skydir=position,
        binsz=config["map"]["size_binsz"],
        width=(config["map"]["size"], config["map"]["size"]),
        proj="CAR", )

    # Now we can define the target dataset with this geometry.
    dataset_stacked = SpectrumDataset.create(
        geom=geom,
        energy_axis_true=energy_axis_true,
        name=f"{source_name}-stacked"
    )


else:
    raise ValueError("Background model is unknown.")

######################################################################
# Find sources SourceCatalogHGPS inside the ROI
# -----------

if config["hgps"]["run"]:
    print_green("Searching for HGPS sources in region ...")
    cat = SourceCatalogHGPS(Path(config["hgps"]["path"]))
    indexes = list(range(len(cat.positions)))

    mask_sources = []
    for index in indexes:
        if position.separation(cat.positions[index]).value < 3:
            mask_sources.append(cat.source_name(index))

    for source in mask_sources:
        try:
            print(source, ',', cat[source].spatial_model(), '\n\n')
        except:
            print(source, ',', cat[source].sky_model(), '\n\n')
else:
    print("Not using HGPS data")

######################################################################
# Create a mask for the region around the source
# -----------

# Create a list of masks (to  potentially add more down the line)
mask_model = []

source_mask = CircleSkyRegion(
    center=position,
    radius=config["source"]["radius"] * u.deg)
mask_model.append(source_mask)

if config["background"]["mask"]["add"]:
    # add another mask before calculating the background

    mask_lon = config["background"]["mask"]["lon"]
    mask_lat = config["background"]["mask"]["lat"]
    mask_frame = config["background"]["mask"]["frame"]
    mask_radius = config["background"]["mask"]["radius"]

    additional_position = SkyCoord(mask_lon, mask_lat,
                                   frame=mask_frame, unit="deg")

    additional_mask = CircleSkyRegion(
        center=additional_position,
        radius=mask_radius * u.deg)

    mask_model.append(additional_mask)


exclusion_mask = wcsgeom.region_mask(regions=mask_model, inside=False)
exclusion_map = exclusion_mask.slice_by_idx({"energy": 0})
exclusion_map.plot()
save_or_show_plot("exclusion-mask")

######################################################################
######################################################################
######################################################################
# Data reduction
# ----------------

print_green("Data selection and reduction ...")

######################################################################
# Create the maker classes to be used
# ----------------

print_green("Create maker classes ...")

if config["background"]["method"] == "fov":
    # Initiate all makers for the Field-of-View background method

    maker = MapDatasetMaker()
    print("Map dataset maker", maker)

    offset_max = 2.0 * u.deg
    maker_safe_mask = SafeMaskMaker(
        methods=["offset-max", "bkg-peak"],
        offset_max=offset_max,
    )

    print("Safe mask maker", maker_safe_mask)

    # Define background maker
    print("Selecting FoV background maker ...")
    maker_bkg = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)


elif config["background"]["method"] == "reflected":
    # Initiate all makers for the reflected background method

    maker = SpectrumDatasetMaker()
    print("Map dataset maker", maker)

    maker_safe_mask = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)
    print("Safe mask maker", maker_safe_mask)

    # Define background maker
    print("Selecting reflected background maker ...")
    maker_bkg = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)


# todo: implement RingBackgroundMaker
# elif user_config["background"]["method"] == "ring":
# maker_bkg = RingBackgroundMaker(r_in="3 deg", width="4 deg", exclusion_mask=exclusion_mask)

else:
    raise ValueError("Background model is unknown.")

try:
    print(maker_bkg)
except AttributeError:
    print("bkg maker does not have print attribute 'print'. Continuing ... ")
    pass

print_yellow(f"Background maker, {type(maker_bkg)}")

######################################################################
# Write the user config to disk for documentation purposes
# ----------------

print_yellow("Writing updated config to disk ... \n")

with open(path / f"config_{std_filename}.log", "w") as config_log:
    pprint(config, config_log, sort_dicts=False)

######################################################################
# Perform data reduction
# ----------------

if config["main"]["data_reduction_run_in_file"]:

    start = time()
    # Get the observations from the previously selected runlist:
    observations = datastore.get_observations(obs_id=config["runlist"]["runlist"],
                                              required_irf=config["datastore"]["required_irf"])

    if config["background"]["method"] == "fov":
        ######################################################################
        # Field-of-View background method
        # ----------------
        print_green("Perform data reduction using 'FoV' background method ...")

        # Run data reduction
        bkg_norms = ["obs_id\tnorm\tnorm_err\ttilt"]
        prb_runs = ["Error with\n"]

        for obs in observations:
            print(f"Processing {obs.obs_id} ...")

            try:
                # First a cutout of the target map is produced
                cutout = dataset_stacked.cutout(
                    obs.get_pointing_icrs(obs.tmid), width=2 * offset_max,
                    name=f"obs-{obs.obs_id}")
                # , mode="partial")
                # galactic

                # A MapDataset is filled in this cutout geometry
                dataset = maker.run(cutout, obs)
                # The data quality cut is applied
                dataset = maker_safe_mask.run(dataset, obs)
                # Fit background model
                dataset = maker_bkg.run(dataset)

                norm_ = dataset.models[0].spectral_model.norm.value
                norm_err = dataset.models[0].spectral_model.norm.error
                tilt = dataset.models[0].spectral_model.tilt.value
                # If the background norm is completely off scale don't stack run
                if np.abs(norm_ - 1.0) > 0.5:
                    # Should usually be norm - 1.0
                    print_yellow(f"Dropping run {obs.obs_id}: Bad norm.")
                if norm_err / norm_ > 0.2:
                    print_yellow(f"Dropping run {obs.obs_id}: Large error on norm.")
                # if np.abs(tilt)>0.5:
                #   print("Dropping run - Bad tilt.")

                bkg_norms.append(
                    f"{obs.obs_id}\t{round(norm_, 5)}\t{round(norm_err, 5)}\t{round(tilt, 5)}")
                # del dataset  # to avoid confusion?
                # del norm_

                # The resulting dataset cutout is stacked onto the final one
                dataset_stacked.stack(dataset)

            except RuntimeError:
                prb_runs.append(obs.obs_id)
                print(f"Error with {obs.obs_id}")
                pass

        # Export background normalisation and problematic runs to file
        bkg_norms = pd.DataFrame(bkg_norms).to_csv(
            path / f"runs_bkg_norms_{std_filename}.txt", index=False, header=False, sep="\n")
        prb_runs = pd.DataFrame(bkg_norms).to_csv(
            path / f"runs_prb_runs_{std_filename}.txt", index=False, header=False, sep="\n")

        ############
        # Save results to file
        # We have one final dataset, which we write to disk and can then print and explore
        # ----------------

        print_yellow("Writing stacked dataset to file ...")
        dataset_stacked.write(path / f"dataset_stacked_{std_filename}.fits.gz", overwrite=True)

        print_yellow(dataset_stacked)
        # del dataset_stacked
        # dataset_stacked = stacked.datasets["stacked"]
        # print(dataset_stacked)



    elif config["background"]["method"] == "reflected":
        ######################################################################
        # Reflected background
        # ----------------

        print_green("Perform data reduction using 'reflected' background method ...")

        # First make sure that warnings (especially RuntimeWarnings which are
        # issued when no OFF region is found) are treated as errors.
        warnings.filterwarnings("error")

        # Initiate counter that tracks the skipped runs (by i.e. not finding OFF region)
        counter_skipped: int = 0

        dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)

        datasets = Datasets()
        for obs in observations:
            print(f"Working on {obs.obs_id} ...")
            try:
                # A SpectrumDataset is filled in this geometry
                dataset = maker.run(dataset_empty.copy(name=f"{obs.obs_id}"), obs)
                # Define safe mask
                dataset = maker_safe_mask.run(dataset, obs)
                # Compute OFF
                dataset_on_off = maker_bkg.run(dataset, obs)
            except RuntimeWarning:
                # Skip runs where no background region can be found
                print(obs.obs_id, "\t is being skipped ...")
                counter_skipped += 1
                continue

            # Append dataset to the list
            datasets.append(dataset)

        # Reset warnings to original state again
        warnings.filterwarnings("default")

        with open(path / "datasets.pickle", "wb") as handle:
            pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

        table_datasets = datasets.info_table()
        table_datasets.write(path / f"datasets_{std_filename}.fits", overwrite=True)
        table_datasets.write(path / f"datasets_{std_filename}.csv",
                             format="csv", delimiter=";", overwrite=True)
        table_datasets.pprint_all()
        counter_analysed = len(table_datasets)

        table_datasets_cumulative = datasets.info_table(cumulative=True)
        table_datasets_cumulative["id"] = table_datasets["name"].astype('int64')
        table_datasets_cumulative["livetime_h"] = table_datasets_cumulative[
                                                      "livetime"] / 3600 * u.hour

        table_datasets_cumulative.write(path / f"datasets_cumulative_{std_filename}.fits",
                                        overwrite=True)
        table_datasets_cumulative.write(path / f"datasets_cumulative_{std_filename}.csv",
                                        format="csv", delimiter=";", overwrite=True)
        table_datasets_cumulative.pprint_all()

    else:
        raise ValueError("Background model is unknown.")

    end = round(((time() - start) / 60), 2)
    print_green(f"Data reduction complete ... \t\t\t Duration:  {end}  min")

if not config["main"]["data_reduction_run_in_file"]:
    ######################################################################
    # This means that the data from a previous run will be read from disk.
    # ----------------

    print_yellow("No data reduction performed. Loading dataset from disk ...")

    if config["background"]["method"] == "fov":

        if config["main"]["dataset_name"] == "nan":
            dataset_stacked = dataset_stacked.read(path / f"dataset_stacked_{std_filename}.fits.gz")
        else:
            dataset_name = config["main"]["dataset_name"]
            print(f"{dataset_name=}")
            dataset_stacked = dataset_stacked.read(path / dataset_name)

    elif config["background"]["method"] == "reflected":

        with open(path / "datasets.pickle", "rb") as handle:
            datasets = pickle.load(handle)

        file_datasets = fits.open(path / f"datasets_{std_filename}.fits")
        table_datasets = table.Table(file_datasets[1].data)

        counter_skipped = np.nan
        counter_analysed = len(table_datasets)

        table_datasets_cumulative = fits.open(path / f"datasets_cumulative_{std_filename}.fits")
        table_datasets_cumulative = table.Table(table_datasets_cumulative[1].data)


    else:
        raise ValueError("Background model is unknown.")

######################################################################
# Plotting and visualisation of results
# ----------------

if config["background"]["method"] == "fov":
    ######################################################################
    # Check the whole dataset (counts, excess, exposure and background)
    # ----------------

    print(dataset_stacked)
    dataset_stacked.mask_fit = None

    dataset_stacked.peek()
    save_or_show_plot("peek")

    ######################################################################
    # Show significance of region under the source mask
    # ---------------

    print(f"Total significance (spectrum dataset) of {source_mask=}:")
    # Calculate total significance
    dataset_component = dataset_stacked.to_spectrum_dataset(on_region=source_mask)
    dataset_component = Datasets(dataset_component)
    info_table = dataset_component.info_table(cumulative=True)
    # print(info_table,"\n",info_table['livetime'][0]/3600.,"\n",info_table['sqrt_ts'][0])
    print_green("Significance = ", round(info_table['sqrt_ts'][0], 3))
    print(info_table.to_pandas())

    excess_estimator = ExcessMapEstimator(correlation_radius='0.2 deg', selection_optional=[])
    excess_maps = excess_estimator.run(dataset_stacked)

    significance_map = excess_maps["sqrt_ts"]
    significance_map.write(path / f"significance_map_{std_filename}.fits", overwrite=True)

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(111, projection=significance_map.geom.wcs)
    # ax2 = plt.subplot(222, projection=excess_map.geom.wcs)

    ax1.set_title("Significance map")
    significance_map.plot(ax=ax1, add_cbar=True, vmin=-6, vmax=6)
    save_or_show_plot("significance-map")

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(111, projection=significance_map.geom.wcs)
    ax1.set_title("Significance map + exclusion mask")
    # plt.colorbar(fraction=0.046, pad=0.04)
    # significance_map.plot(ax=ax1, add_cbar=True, cbarfraction=0.046, vmin=-6, vmax=6)
    significance_map.plot(ax=ax1, add_cbar=True, vmin=-6, vmax=6)
    exclusion_map.plot(ax=ax1, add_cbar=False, alpha=0.3)
    save_or_show_plot("significance-map-excl-mask")

    ######################################################################
    # Plot histogram of significance
    # ----------------

    # create a 2D mask for the images
    mask_image = exclusion_mask.sum_over_axes()
    mask_image.data = mask_image.data.astype(bool)

    significance_map_off = significance_map * mask_image
    significance_all = significance_map.data[np.isfinite(significance_map.data)]
    significance_off = significance_map_off.data[
        np.isfinite(significance_map_off.data)
    ]

    plt.hist(significance_all, density=True, alpha=0.5,
             color="red", label="all bins", bins=21)

    plt.hist(significance_off, density=True, alpha=0.5,
             color="blue", label="off bins", bins=21)

    # Now, fit the off distribution with a Gaussian
    mu, std = norm.fit(significance_off)
    x = np.linspace(-8, 8, 50)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, lw=2, color="black")
    plt.legend()

    plt.xlabel("Significance")
    plt.xlim(-6, 6)

    plt.yscale("log")
    plt.ylim(1e-5, 1)

    # x_min, x_max = np.min(significance_all), np.max(significance_all)

    plt.text(0, 1e-3, f"mu:  {mu:.2f} \n std:  {std:.2f}  ")
    print(f"Fit results: mu:  {mu:.2f}, std:    {std:.2f}")

    save_or_show_plot("histogram")

    ######################################################################
    # Plot a smooth counts map
    # ----------------

    dataset_stacked.counts.sum_over_axes().smooth("0.03 deg").plot(stretch="sqrt", add_cbar=True)
    # dataset_stacked.counts.smooth(0.02 * u.deg).plot_interactive(add_cbar=True)
    plt.title("Counts")
    save_or_show_plot("counts-map")

    ######################################################################
    # Plot the excess image
    # ----------------

    excess = dataset_stacked.excess.sum_over_axes().smooth("0.15 deg")
    excess.plot(stretch="linear", add_cbar=True)  # "linear"
    plt.title("Excess counts")
    save_or_show_plot("excess-map")

    dataset_stacked.write(path / f"dataset_stacked_{std_filename}.fits.gz", overwrite=True)

    ######################################################################
    # Plot the background map
    # ----------------

    background = dataset_stacked.background.sum_over_axes().smooth("0.03 deg")
    background.plot(stretch="linear", add_cbar=True)
    # dataset_stacked.counts.smooth(0.02 * u.deg).plot_interactive(add_cbar=True)
    plt.title("Background map")
    save_or_show_plot("bkg-map")

    ######################################################################
    # We can check the PSF
    # ----------------

    psf = dataset_stacked.psf.peek()
    save_or_show_plot("psf")

    # fig, ((cont_rad, psf_at_centre), (psf_exposure, cont_rad_1TeV)) = plt.subplots(2, 2)
    # cont_rad = dataset_stacked.psf.plot_containment_radius_vs_energy()
    # psf_at_center = dataset_stacked.psf.plot_psf_vs_rad(energy_true=energy_axis_true)
    # psf_exposure = dataset_stacked.psf.exposure_map.sum_over_axes()
    # cont_rad_1TeV = dataset_stacked.psf.containment_radius_map().sum_over_axes()

    ######################################################################
    # Check energy dispersion in the center of the map
    # ----------------

    # dataset_stacked.edisp.peek()
    edisp = dataset_stacked.edisp.peek()
    save_or_show_plot("edisp")



elif config["background"]["method"] == "reflected":

    print("Creating plots ...")

    ######################################################################
    # Locate the yearly limits of all the runs to visualise in livetime plots
    # ----------------

    # Last run number on 2020-12-27: 164384
    # Close to last on 2021-12-31: 172067
    # Last on 2022-12-30: 180062
    # Last on 2023-12-31: 187032

    df = table_datasets_cumulative.to_pandas()
    datasets_2021 = df[df["id"].between(164384, 172067)]
    datasets_2022 = df[df["id"].between(172067, 180062)]
    datasets_2023 = df[df["id"].between(180062, 187032)]
    del df

    max_datasets_2021 = float(datasets_2021["livetime_h"].max())
    max_datasets_2022 = float(datasets_2022["livetime_h"].max())
    max_datasets_2023 = float(datasets_2023["livetime_h"].max())

    list_max_datasets = [[max_datasets_2021, "2021"],
                         [max_datasets_2022, "2022"],
                         [max_datasets_2023, "2023"]]

    # Create boolean list of years that actually contain at least one element
    good_years = [not np.isnan(element[0]) for element in list_max_datasets]
    # Apply the boolean filter to the relevant list
    list_max_datasets = list(compress(list_max_datasets, good_years))


    def plt_year_limits(max_values: list, ymin: float, ymax: float):

        for item in max_values:
            plt_line(item[0], item[1], ymin, ymax)
        return 0


    def plt_line(max_value: float, year: str, ymin: float, ymax: float):

        color = colour_list[2]
        text_ypos = ymax / 3

        plt.text(max_value, text_ypos,
                 f"<-- {year}|", color=color, horizontalalignment='right')
        plt.vlines([max_value + 0.1], ymin, ymax, colors=color, ls='--')
        return 0


    ######################################################################
    # Produce livetime plots
    # ----------------

    # Plot SQRT(TS) over livetime
    plt.xlabel("Livetime (h)")
    plt.ylabel("Sqrt(TS)")
    plt.xlim(0, table_datasets_cumulative["livetime_h"].max() + 0.5)

    plt.title(f"Sqrt(TS) over time ({source_name})")
    plt.plot(table_datasets_cumulative["livetime_h"], table_datasets_cumulative["sqrt_ts"],
             marker="o")
    plt_year_limits(list_max_datasets,
                    float(table_datasets_cumulative["sqrt_ts"].min()),
                    float(table_datasets_cumulative["sqrt_ts"].max()))

    save_or_show_plot("sqrt-ts-over-time")

    # Plot Excess counts over livetime
    plt.xlabel("Livetime (h)")
    plt.ylabel("Excess counts")
    plt.xlim(0, table_datasets_cumulative["livetime_h"].max() + 0.5)

    plt.title(f"Excess counts over time ({source_name})")
    plt.plot(table_datasets_cumulative["livetime_h"], table_datasets_cumulative["excess"],
             marker="o")
    plt_year_limits(list_max_datasets,
                    float(table_datasets_cumulative["excess"].min()),
                    float(table_datasets_cumulative["excess"].max()))

    plt.text(0.2, 50,
             f"Total runs:    {counter_observations}\n"
             f"    Analysed:  {counter_analysed}\n"
             f"    Skipped:    {counter_skipped}")

    save_or_show_plot("excess-over-time")

    # Plot Background counts over livetime
    plt.xlabel("Livetime (h)")
    plt.ylabel("Background counts and excess counts")
    plt.xlim(0, table_datasets_cumulative["livetime_h"].max() + 0.5)

    plt.title(f"Background and excess counts over time ({source_name})")
    plt.plot(table_datasets_cumulative["livetime_h"], table_datasets_cumulative["excess"],
             marker="o", label="Excess counts")
    plt.plot(table_datasets_cumulative["livetime_h"], table_datasets_cumulative["background"],
             marker="o", label="Background counts")

    plt_year_limits(list_max_datasets,
                    float(table_datasets_cumulative["excess"].min()),
                    float(table_datasets_cumulative["excess"].max()))

    plt_year_limits(list_max_datasets,
                    float(table_datasets_cumulative["background"].min()),
                    float(table_datasets_cumulative["background"].max()))

    plt.legend()
    save_or_show_plot("background-over-time")


######################################################################
######################################################################
######################################################################
# Modeling and fitting
# ----------------
#
# Now comes the interesting part of the analysis - choosing appropriate
# models for our source and fitting them.


######################################################################
# To perform the fit on a restricted energy range, we can create a
# specific *mask*. On the dataset, the `mask_fit` is a `Map` sharing
# the same geometry as the `MapDataset` and containing boolean data.
#
# To create a mask to limit the fit within a restricted energy range, one
# can rely on the `~gammapy.maps.Geom.energy_mask()` method.
#
# For more details on masks and the techniques to create them in gammapy,
# please check out the dedicated :doc:`/tutorials/api/mask_maps` tutorial.
#
# print_green("\n\nExecuting modelling section ... ")
#
# dataset_stacked.mask_fit = dataset_stacked.counts.wcsgeom.energy_mask(
#     energy_min=0.3 * u.TeV, energy_max=None
# )


if config["model"]["run"]:

    if config["background"]["method"] == "fov":
        # Perform fitting for Field-of-View Method

        if config["model"]["spatial"]["frame"] == "galactic":
            initial_fit_position = position.galactic
            initial_fit_radius = radius
        else:
            raise NotImplementedError("Please use galactic coordinates for this.")


        config_models = [config["model0"]]
        try:
            config_models.append(config["model1"])
        except KeyError:
            print_yellow("'model2' does not exist. Continuing ...")

        def set_spatial_position_params(config_model: dict):
            """
            Function to set the spatial parameters from the config, such as
            positional arguments and spatial search limits.

            :param config_model: dict, containing parameters of one of the models
            :return:
            """

            lon_limit: list = config_model["spatial"]["lon"]["limit"]
            lat_limit: list = config_model["spatial"]["lat"]["limit"]

            # if the limits are not "nan", i.e. given, set them to limit the
            # search parameters of the optimiser
            if lon_limit != ["nan", "nan"]:
                spatial_model.lon_0.min = lon_limit[0]
                spatial_model.lon_0.max = lon_limit[1]

            if lon_limit != ["nan", "nan"]:
                spatial_model.lat_0.min = lat_limit[0]
                spatial_model.lat_0.max = lat_limit[1]


            try:
                radius_limit: list = config_model["spatial"]["radius"]["limit"]
                if radius_limit != ["nan", "nan"]:
                    spatial_model.r_0.min = radius_limit[0]
                    spatial_model.r_0.max = radius_limit[1]

            except KeyError:
                print_red("Radius parameter limits not given. Continuing ...")
                radius_limit = ["nan", "nan"]

            return f"{lon_limit=}, {lat_limit=}, {radius_limit=}"

        # Create list for the sky models
        models = []

        for config_model in config_models:

            lon = config_model["spatial"]["lon"]["value"]
            lat = config_model["spatial"]["lat"]["value"]

            # If the source position values are not given, set the default
            # values from the config used initally.
            if lon == "default":
                lon = initial_fit_position.l.value
            if lat == "default":
                lat = initial_fit_position.b.value

            # Check if radius is given and handle exception if not
            try:
                r = config_model["spatial"]["radius"]["value"]
                if r == "default":
                    r = radius
            except KeyError:
                print("No parameter 'radius' given. Continuing ... ")


            # Now chose the spatial model
            spatial_model = config_model["spatial"]["name"]
            if spatial_model == "PointSpatialModel":
                spatial_model = PointSpatialModel(
                    lon_0=lon * u.deg,
                    lat_0=lat * u.deg,
                    frame="galactic",
                    # amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
                    # reference=1 * u.TeV,
                )


                set_spatial_position_params(config_model)


            elif spatial_model == "DiskSpatialModel":
                spatial_model = DiskSpatialModel(
                    lon_0=lon * u.deg,
                    lat_0=lat * u.deg,
                    frame="galactic",
                    r_0=r * u.deg
                )

                set_spatial_position_params(config_model)

                try:
                    spatial_model.lon_0.frozen = config_model["spatial"]["lon"]["frozen"]
                    spatial_model.lat_0.frozen = config_model["spatial"]["lat"]["frozen"]
                    spatial_model.r_0.frozen = config_model["spatial"]["radius"]["frozen"]
                except KeyError:
                    print_yellow("unknown parameters set to frozen in spatial model.")


            elif spatial_model == "GaussianSpatialModel":
                raise NotImplementedError("The GaussianSpatialModel has not"
                                          "been fully implemented yet.")

            else:
                raise NotImplementedError("Spatial model type not supported, please chose one of"
                                "the following:"
                                "'PointSpatialModel',"
                                "'DiskSpatialModel'.")

            # Define the spectral model
            spectral_model = config_model["spectral"]["name"]

            if spectral_model == "ExpCutoffPowerLawSpectralModel":
                spectral_model = ExpCutoffPowerLawSpectralModel(
                    index=2,
                    amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
                    lambda_=0.1 * u.Unit("TeV-1"),
                    reference=1 * u.TeV,
                )

            elif spectral_model == "PowerLawSpectralModel":
                spectral_model = PowerLawSpectralModel(
                    index=2,
                    amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
                    lambda_=0.1 * u.Unit("TeV-1"),
                    reference=1 * u.TeV,
                )

            elif spectral_model == "LogParabolaSpectralModel":
                spectral_model = LogParabolaSpectralModel(
                    alpha=2,
                    beta=1,
                    amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
                    reference=1 * u.TeV,
                )

            else:
                raise Exception("Spectral model type not supported, please chose one of"
                                "the following:"
                                "'PowerLawSpectralModel',"
                                "'ExpCutoffPowerLawSpectralModel',"
                                "'LogParabolaSpectralModel'. ")

            # Print chosen models
            print_green(f"Spatial and spectral model for {config_model=}")
            print(spatial_model, 2 * "\n", spectral_model)

            # Define the sky model

            name = config_model["name"]
            if name == "default":
                name = source_name

            sky_model = SkyModel(spatial_model=spatial_model,
                                 spectral_model=spectral_model,
                                 name=name)
            models.append(sky_model)
            print(f"{sky_model=}")


        # Assign a background model
        bkg_model = FoVBackgroundModel(dataset_name=f"{source_name}-stacked")
        models.append(bkg_model)

        # Add models to the stacked dataset
        # dataset_stacked.models = [sky_model, bkg_model]
        dataset_stacked.models = models

        # todo potential snipped to use finding peaks
        # from gammapy.estimators import TSMapEstimator
        # from gammapy.modeling.models import DiskSpatialModel
        # from gammapy.estimators.utils import find_peaks
        #
        # dataset_image = dataset_stacked.to_image()
        # geom_image = dataset_image.geoms["geom"]
        #
        # show_plots = True
        #
        # fig, (ax1, ax2, ax3) = plt.subplots(
        #     figsize=(15, 5),
        #     ncols=3,
        #     subplot_kw={"projection": geom_image.wcs},
        #     gridspec_kw={"left": 0.1, "right": 0.9},
        # )
        #
        # ax1.set_title("Counts map")
        # dataset_image.counts.smooth(2).plot(ax=ax1, vmax=8)
        #
        # ax2.set_title("Background map")
        # dataset_image.background.plot(ax=ax2, vmax=8)
        #
        # ax3.set_title("Excess map")
        # dataset_image.excess.smooth(3).plot(ax=ax3, vmax=2)
        # plt.show()
        #
        # spatial_model = GaussianSpatialModel(sigma="0.05 deg")
        # spectral_model = PowerLawSpectralModel(index=2)
        # model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
        #
        # ts_image_estimator = TSMapEstimator(
        #     model,
        #     kernel_width="0.5 deg",
        #     selection_optional=[],
        #     downsampling_factor=2,
        #     sum_over_energy_groups=False,
        #     energy_edges=[0.1, 10] * u.TeV,
        # )
        #
        # images_ts = ts_image_estimator.run(dataset_stacked)
        #
        # sources = find_peaks(
        #     images_ts["sqrt_ts"],
        #     threshold=3.5,
        #     min_distance="0.2 deg",
        # )
        # display(sources)

        ######################################################################
        # Now we can run the fit
        # ----------------
        print("\nStarting fitting ...")

        # Start timing
        start = time()

        # Start fitting
        # fit = Fit(optimize_opts={"print_level": 1})
        fit = Fit(backend=config["model"]["backend"])
        result = fit.run(datasets=[dataset_stacked])
        print(result)

        # End timing
        end = round(((time() - start) / 60), 2)
        print_green(f"Fitting complete ... \t\t\t Duration:  {end}  min")

        print(dataset_stacked)

        # Explore the fit results
        # First the fitted parameters values and their errors.
        table_fit = dataset_stacked.models.to_parameters_table()
        table_fit.pprint_all()

        table_fit.write(path / f"fit-results_{std_filename}.fits", overwrite=True)

        # Print the position of the resulting fit
        pprint(table_fit[3:6])
        # Create Skycoord object from fit position (and convert to Galactic coordinates)
        fit_pos = SkyCoord(round(table_fit[3]["value"], 4), round(table_fit[4]["value"], 4),
                           unit="deg", frame="galactic")

        # Create Skyregion with the Galactic position as centre
        fit_region = CircleSkyRegion(center=fit_pos,
                                     radius=round(table_fit[5]["value"], 3) * u.deg)
        print_green(f"{fit_region=}")

        try:
            pprint(table_fit[12:15])
            fit_pos2 = SkyCoord(round(table_fit[12]["value"], 4), round(table_fit[13]["value"], 4),
                           unit="deg", frame="galactic")
            # Create Skyregion with the Galactic position as centre
            fit_region2 = CircleSkyRegion(center=fit_pos2,
                                         radius=round(table_fit[14]["value"], 3) * u.deg)
            print_green(f"{fit_region=}")
        except IndexError:
            print_yellow("Index out of range. Probably only one source model given as input. ")


        # todo: this part is a copy of what is shown above before fitting
        ######################################################################
        # Show significance of region under the source mask
        # ---------------

        print(f"Total significance (spectrum dataset) of {fit_region=}:")
        # Calculate total significance
        dataset_component = dataset_stacked.to_spectrum_dataset(on_region=fit_region)
        dataset_component = Datasets(dataset_component)
        info_table = dataset_component.info_table(cumulative=True)
        print_green("Significance = ", round(info_table['sqrt_ts'][0], 3))
        print(info_table.to_pandas())

        try:
            print(f"Total significance (spectrum dataset) of {fit_region2=}:")
            # Calculate total significance
            dataset_component = dataset_stacked.to_spectrum_dataset(on_region=fit_region2)
            dataset_component = Datasets(dataset_component)
            info_table = dataset_component.info_table(cumulative=True)
            print_green("Significance = ", round(info_table['sqrt_ts'][0], 3))
            print(info_table.to_pandas())
        except NameError:
            print_yellow("Only one fit region given. Continuing ...")

        # Assign the fitted model
        # reduced.models = model

        # Here we can plot the number of predicted counts for each model
        # and for the background in our dataset.In order to do this, we can use
        # the plot_npred_signal function.

        # This creates the plot

        try:
            plot_npred_signal(dataset_stacked)
            # This saves the plot
            save_or_show_plot("npred-signal")
        except:
            print_red("There was an issue with the 'npred-signal' plotting")

        # Inspecting residuals
        # For any fit it is useful to inspect the residual images. We have a
        # few options on the dataset object to handle this.First we can use
        # plot_residuals_spatial to plot a residual image, summed over all
        # energies:

        dataset_stacked.plot_residuals_spatial(method="diff", vmin=-0.5, vmax=0.5)
        save_or_show_plot("residuals-spatial")

        # In addition, we can also specify a region in the map to show the spectral residuals:

        # region = CircleSkyRegion(center=SkyCoord("83.63 deg", "22.14 deg"), radius=0.5 * u.deg)

        dataset_stacked.plot_residuals(
            kwargs_spatial=dict(method="diff/sqrt(model)", vmin=-0.5, vmax=0.5),
            kwargs_spectral=dict(region=fit_region),
        )
        save_or_show_plot("residuals-spatial-mask")

        dataset_stacked.plot_residuals(
            kwargs_spatial=dict(method="diff", vmin=-1, vmax=1),
            kwargs_spectral=dict(region=fit_region),
        )
        save_or_show_plot("residuals-spatial-mask")


        # Making a butterfly plot The SpectralModel component can be used to
        # produce a so - called, butterfly plot showing the envelope of the
        # model taking into account parameter uncertainties:

        # spec = sky_model.spectral_model
        spec = models[0].spectral_model

        energy_bounds = [0.3, 100] * u.TeV
        spec.plot(energy_bounds=energy_bounds, energy_power=2)
        ax = spec.plot_error(energy_bounds=energy_bounds, energy_power=2)
        save_or_show_plot("spectrum-butterfly")

        # Computing flux points
        # We can now compute some flux points using the FluxPointsEstimator.
        # Besides the list of datasets to use, we must provide it the energy
        # intervals on which to compute flux points as well as the model
        # component name.

        energy_limit = config["model"]["sed"]["energy_limit"]
        energy_nbins: int = config["model"]["sed"]["energy_nbins"]
        energy_min, energy_max = energy_limit[0], energy_limit[1]
        energy_edges = np.geomspace(energy_min, energy_max, energy_nbins) * u.TeV

        # # Now we create an instance of the FluxPointsEstimator, by passing the dataset
        # # and the energy binning:

        fpe = FluxPointsEstimator(
            energy_edges=energy_edges, source=f"{source_name}", selection_optional="all"
        )
        flux_points = fpe.run(datasets=[dataset_stacked])

        # Here is the table of the resulting flux points:

        table_sed = flux_points.to_table(sed_type="dnde", formatted=True)
        table_sed.pprint_all()  # show entire table no matter how long, wide, etc with units
        table_sed.write(path / f"sed-flux_{std_filename}.fits", overwrite=True)

        ax = spec.plot_error(energy_bounds=energy_bounds, sed_type="e2dnde")
        flux_points.plot(ax=ax, sed_type="e2dnde", color=colour_list[1])

        # Overplot model and butterfly error
        spec.plot(energy_bounds=energy_bounds, sed_type="e2dnde",color=colour_list[1], linestyle="--")
        spec.plot_error(energy_bounds=energy_bounds, sed_type="e2dnde", color=colour_list[1])

        # flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde")
        save_or_show_plot("sed-flux")


        ######################################################################
        # Replot the significance map, but this time with the best-fit result
        # ----------------

        significance_map = excess_maps["sqrt_ts"]
        # significance_map.write(path / f"significance_map_{std_filename}.fits", overwrite=True)

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(111, projection=significance_map.geom.wcs)

        ax1.set_title("Significance map + best-fit")
        significance_map.plot(ax=ax1, add_cbar=True, vmin=-5, vmax=5, cmap="coolwarm")

        # Add contours of significance (3-5 sigma)
        sign_red = significance_map.reduce('energy')
        ax1.contour(sign_red, levels=[3, 4, 5], colors=['#3a5a40', '#558157', '#5a9f68'])
        contours_patch = mpatches.Patch(color="#558157",
                                        label=r"Significance contours at 3, 4, 5 $\sigma$",
                                        fill=None, alpha=0.8,
                                        linestyle="solid")

        # First, show the intial position that the fit was started from
        radius_colour = "xkcd:grey"
        initial_fit_pos = SphericalCircle(position, radius * u.deg,
                                          alpha=0.8,
                                          edgecolor=radius_colour, facecolor='none',
                                          transform=ax1.get_transform('galactic'),
                                          linewidth=2.6, linestyle='dashed')
        ax1.add_patch(initial_fit_pos)
        ax1.text(position.l.value, position.b.value,
                 s="+", alpha=0.8,
                 verticalalignment='center', horizontalalignment='center',
                 transform=ax1.get_transform('galactic'),
                 fontweight='bold', fontsize=18, color=radius_colour)

        initial_fit_pos_patch = mpatches.Patch(color=radius_colour,
                                               label=f"Position before fitting "
                                                     f"({initial_fit_position.l.value}, "
                                                     f"{initial_fit_position.b.value},"
                                                     f" r={initial_fit_radius})",
                                               fill=None, alpha=0.8,
                                               linestyle="dashed")

        # Then, show the best-fit position that was found
        radius_colour = "xkcd:teal blue"
        best_fit_pos = SphericalCircle(fit_pos, fit_region.radius, alpha=0.8,
                                       edgecolor=radius_colour, facecolor='none',
                                       transform=ax1.get_transform('galactic'),
                                       linewidth=2.6, linestyle='dashed')
        ax1.add_patch(best_fit_pos)
        ax1.text(fit_pos.l.value, fit_pos.b.value,
                 s="+", alpha=0.8,
                 verticalalignment='center', horizontalalignment='center',
                 transform=ax1.get_transform('galactic'),
                 fontweight='bold', fontsize=18, color=radius_colour)

        fit_patch = mpatches.Patch(color=radius_colour,
                                   label=f"Best-fit position ({fit_pos.l.value}, "
                                         f"{fit_pos.b.value}, "
                                         f"r={fit_region.radius})",
                                   fill=None, alpha=0.8,
                                   linestyle="dashed")

        # Adding position of second fitting region
        try:
            radius_colour = "xkcd:irish green"
            best_fit_pos2 = SphericalCircle(fit_pos2, fit_region2.radius, alpha=0.8,
                                           edgecolor=radius_colour, facecolor='none',
                                           transform=ax1.get_transform('galactic'),
                                           linewidth=2.6, linestyle='dashed')
            ax1.add_patch(best_fit_pos2)
            ax1.text(fit_pos2.l.value, fit_pos2.b.value,
                     s="+", alpha=0.8,
                     verticalalignment='center', horizontalalignment='center',
                     transform=ax1.get_transform('galactic'),
                     fontweight='bold', fontsize=18, color=radius_colour)

            fit_patch2 = mpatches.Patch(color=radius_colour,
                                       label=f"Best-fit position 2 ({fit_pos2.l.value}, "
                                             f"{fit_pos2.b.value}, "
                                             f"{fit_region2.radius})",
                                       fill=None, alpha=0.8,
                                       linestyle="dashed")

            plt.legend(handles=[initial_fit_pos_patch, fit_patch, fit_patch2, contours_patch])

        except NameError:
            print_yellow("Only one fit region given. Continuing ...")
            plt.legend(handles=[initial_fit_pos_patch, fit_patch, contours_patch])


        save_or_show_plot("significance-map-best-fit")



        ######################################################################
        # Plot the residual map, but this time with the best-fit result
        # ----------------

        excess_estimator = ExcessMapEstimator(correlation_radius='0.2 deg', selection_optional=[])
        excess_maps = excess_estimator.run(dataset_stacked)

        resid_map = excess_maps["sqrt_ts"]
        resid_map.write(path / f"resid_map_{std_filename}.fits", overwrite=True)

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(111, projection=resid_map.geom.wcs)

        ax1.set_title("Residual map + best-fit")
        resid_map.plot(ax=ax1, add_cbar=True, vmin=-5, vmax=5, cmap="coolwarm")

        resid_reduced = resid_map.reduce('energy')
        ax1.contour(resid_reduced, levels=[3, 4, 5], colors=['#3a5a40', '#558157', '#5a9f68'])
        contours_patch = mpatches.Patch(color='#558157',
                                               label=r"Significance contours at 3, 4, 5 $\sigma$",
                                               fill=None, alpha=0.8,
                                               linestyle="solid")
        # First, show the intial position that the fit was started from
        radius_colour = "xkcd:grey"
        initial_fit_pos = SphericalCircle(position, radius * u.deg,
                                          alpha=0.8,
                                          edgecolor=radius_colour, facecolor='none',
                                          transform=ax1.get_transform('galactic'),
                                          linewidth=2.6, linestyle='dashed')
        ax1.add_patch(initial_fit_pos)
        ax1.text(position.l.value, position.b.value,
                 s="+", alpha=0.8,
                 verticalalignment='center', horizontalalignment='center',
                 transform=ax1.get_transform('galactic'),
                 fontweight='bold', fontsize=18, color=radius_colour)

        initial_fit_pos_patch = mpatches.Patch(color=radius_colour,
                                               label=f"Position before fitting "
                                                     f"({initial_fit_position.l.value}, "
                                                     f"{initial_fit_position.b.value},"
                                                     f" r={initial_fit_radius})",
                                               fill=None, alpha=0.8,
                                               linestyle="dashed")

        # Then, show the best-fit position that was found
        radius_colour = "xkcd:teal blue"
        best_fit_pos = SphericalCircle(fit_pos, fit_region.radius, alpha=0.8,
                                       edgecolor=radius_colour, facecolor='none',
                                       transform=ax1.get_transform('galactic'),
                                       linewidth=2.6, linestyle='dashed')
        ax1.add_patch(best_fit_pos)
        ax1.text(fit_pos.l.value, fit_pos.b.value,
                 s="+", alpha=0.8,
                 verticalalignment='center', horizontalalignment='center',
                 transform=ax1.get_transform('galactic'),
                 fontweight='bold', fontsize=18, color=radius_colour)

        fit_patch = mpatches.Patch(color=radius_colour,
                                   label=f"Best-fit position ({fit_pos.l.value}, "
                                         f"{fit_pos.b.value}, "
                                         f"r={fit_region.radius})",
                                   fill=None, alpha=0.8,
                                   linestyle="dashed")

        # Adding position of second fitting region
        try:
            radius_colour = "xkcd:irish green"
            best_fit_pos2 = SphericalCircle(fit_pos2, fit_region2.radius, alpha=0.8,
                                            edgecolor=radius_colour, facecolor='none',
                                            transform=ax1.get_transform('galactic'),
                                            linewidth=2.6, linestyle='dashed')
            ax1.add_patch(best_fit_pos2)
            ax1.text(fit_pos2.l.value, fit_pos2.b.value,
                     s="+", alpha=0.8,
                     verticalalignment='center', horizontalalignment='center',
                     transform=ax1.get_transform('galactic'),
                     fontweight='bold', fontsize=18, color=radius_colour)

            fit_patch2 = mpatches.Patch(color=radius_colour,
                                        label=f"Best-fit position 2 ({fit_pos2.l.value}, "
                                              f"{fit_pos2.b.value}, "
                                              f"{fit_region2.radius})",
                                        fill=None, alpha=0.8,
                                        linestyle="dashed")

            plt.legend(handles=[initial_fit_pos_patch, fit_patch, fit_patch2, contours_patch])

        except NameError:
            print_yellow("Only one fit region given. Continuing ...")
            plt.legend(handles=[initial_fit_pos_patch, fit_patch, contours_patch])

        save_or_show_plot("resid-map-best-fit")


        print_green("--- END OF FOV ---")


    elif config["background"]["method"] == "reflected":

        # -------------------------------
        # Perform spectral model fitting
        # Here we perform a joint fit.
        # We first create the model, here a simple powerlaw, and assign it to every
        # dataset in the Datasets.

        print("\nStarting fitting ...")
        start = time()

        print_yellow("This method only uses the specified spectral model."
                     "The spatial model given in the config will be ignored.")

        spectral_model = config["model"]["spectral"]["name"]

        if spectral_model == "ExpCutoffPowerLawSpectralModel":
            spectral_model = ExpCutoffPowerLawSpectralModel(
                index=2,
                amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
                lambda_=0.1 * u.Unit("TeV-1"),
                reference=1 * u.TeV,
            )

        elif spectral_model == "PowerLawSpectralModel":
            spectral_model = PowerLawSpectralModel(
                index=2,
                amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
                lambda_=0.1 * u.Unit("TeV-1"),
                reference=1 * u.TeV,
            )

        elif spectral_model == "LogParabolaSpectralModel":
            spectral_model = LogParabolaSpectralModel(
                alpha=2,
                beta=1,
                amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"),
                reference=1 * u.TeV,
            )

        else:
            raise Exception("Spectral model type not supported, please chose one of"
                            "the following:"
                            "'PowerLawSpectralModel',"
                            "'ExpCutoffPowerLawSpectralModel',"
                            "'LogParabolaSpectralModel'. ")

        sky_model = SkyModel(spectral_model=spectral_model, name=f"{source_name}")

        datasets.models = [sky_model]

        ######################################################################
        # Now we can run the fit
        # ----------------

        fit_joint = Fit()
        result_joint = fit_joint.run(datasets=datasets)
        print(result_joint)

        end = round(((time() - start) / 60), 2)
        print_green(f"Fitting complete ... \t\t\t Duration:  {end}  min")

        # copy to compare later
        model_best_joint = sky_model.copy()

        # Explore the fit results
        # First the fitted parameters values and their errors.
        table_fit = datasets.models.to_parameters_table()
        print(table_fit)

        table_fit.write(path / f"fit-results_{std_filename}.fits", overwrite=True)

        # First stack them all
        reduced = datasets.stack_reduce()

        # Assign the fitted model
        reduced.models = sky_model

        # Plot the result
        ax_spectrum, ax_residuals = reduced.plot_fit()
        reduced.plot_masks(ax=ax_spectrum)

        save_or_show_plot("sed_counts")

        # # Plot off regions
        #
        # plt.figure()
        # ax = exclusion_mask.plot()
        # geom.coord_to_pix(ax.wcs).plot(ax=ax, edgecolor="k")
        # plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)
        # plt.show()
        # print("exit")

        # Compute Flux Points
        # To round up our analysis we can compute flux points by fitting the norm
        # of the global model in energy bands. We’ll use a fixed energy binning for now:

        e_min, e_max = 0.7, 30
        # e_min, e_max = config["energy"]["true_low"], config["energy"]["true_high"]
        energy_edges = np.geomspace(e_min, e_max, 11) * u.TeV

        # Now we create an instance of the FluxPointsEstimator, by passing the dataset
        # and the energy binning:
        from gammapy.estimators import FluxPointsEstimator

        fpe = FluxPointsEstimator(
            energy_edges=energy_edges, source=f"{source_name}", selection_optional="all"
        )
        flux_points = fpe.run(datasets=datasets)

        # Here is the table of the resulting flux points:

        table_sed = flux_points.to_table(sed_type="dnde", formatted=True)
        table_sed.pprint_all()  # show entire table no matter how long, wide, etc with units
        table_sed.write(path / f"sed-flux_{std_filename}.fits", overwrite=True)

        ax = geom.wcs

        fig, ax = plt.subplots()
        flux_points.plot(ax=ax, sed_type="e2dnde", color=colour_list[2])
        flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde")

        save_or_show_plot("sed-flux")

        print_green("--- END OF REFLECTED ---")


######################################################################
# Print runtime, save log and exit
# ----------------

# End timing
end_total = round(((time() - start_total) / 60), 2)
print_green(f"\n\nScript runtime duration:  {end_total}  min")

# Save log
if config["main"]["create_logfile"]:
    sys.stdout.log.close()

sys.exit(0)

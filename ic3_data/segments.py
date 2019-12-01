from __future__ import print_function, division
import numpy as np

from icecube import icetray

from ic3_data.container import DNNDataContainer
from ic3_data.data import DNNContainerHandler


@icetray.traysegment
def CreateDNNData(tray, name,
                  NumDataBins,
                  RelativeTimeMethod,
                  DataFormat,
                  PulseKey=None,
                  DOMExclusions=None,
                  PartialExclusion=None,
                  TimeBins=None,
                  TimeQuantiles=None,
                  AutoencoderSettings=None,
                  AutoencoderEncoderName=None,
                  CascadeKey='MCCascade',
                  OutputKey='dnn_data',
                  Verbose=True,
                  ):

    # create DNN data container object
    container = DNNDataContainer()

    # configure container
    container.configure(num_bins=NumDataBins,
                        relative_time_method=RelativeTimeMethod,
                        data_format=DataFormat,
                        pulse_key=PulseKey,
                        time_bins=TimeBins,
                        time_quantiles=TimeQuantiles,
                        autoencoder_settings=AutoencoderSettings,
                        autoencoder_name=AutoencoderEncoderName,
                        cascade_key=CascadeKey,
                        dom_exclusions=DOMExclusions,
                        partial_exclusion=PartialExclusion,
                        )

    # set up container
    container.set_up()

    tray.AddModule(DNNContainerHandler, name + '_DNNContainerHandler',
                   DNNDataContainer=container,
                   OutputKey=OutputKey,
                   Verbose=Verbose)

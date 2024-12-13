#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import sys

sys.path.append("/opt/nvidia/deepstream/deepstream-7.0/sources/deepstream_python_apps/apps")
import gi
import configparser
import argparse

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import os
import math
import platform
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA

import pyds

no_display = False
silent = False
file_loop = False
perf_data = None

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 540
MUXER_OUTPUT_HEIGHT = 540  # 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 640  # 1280
TILED_OUTPUT_HEIGHT = 360  # 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1

class_names = []
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

frame_count = 0
start_time = time.time()

def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    The function pgie_src_pad_buffer_probe() is a callback function that is called every time a buffer
    is received on the source pad of the pgie element. 
    The function calculate the batch metadata from the buffer and iterates through the list of frame
    metadata in the batch. 
    For each frame, it iterates through the list of object metadata and prints the frame number, number
    of objects detected, and the number of vehicles, persons, bicycles, and road signs detected in the
    frame. 
    The function also retrieves the frame rate of the stream from the frame metadata
    :param pad: The pad on which the probe is attached
    :param info: The Gst.PadProbeInfo object that contains the buffer
    :param u_data: User data passed to the probe
    :return: The return value is a Gst.PadProbeReturn.OK.
    """
    global frame_count, start_time

    frame_count += 1
    elapsed_time = time.time() - start_time

    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        display_meta.text_params[0].display_text = f"FPS: {fps:.2f}"
        display_meta.text_params[0].x_offset = 10  # X-coordinate of the text
        display_meta.text_params[0].y_offset = 12  # Y-coordinate of the text
        display_meta.text_params[0].font_params.font_name = "Serif"
        display_meta.text_params[0].font_params.font_size = 12
        display_meta.text_params[0].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)  # RGBA White
        display_meta.text_params[0].set_bg_clr = 1
        display_meta.text_params[0].text_bg_clr.set(0.0, 0.0, 0.0, 0.5)  # RGBA Black background
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            object_id = obj_meta.object_id 
            print(f"Object ID: {object_id}, Class ID: {obj_meta.class_id} ({class_names[obj_meta.class_id]})")
            
            source_id = frame_meta.source_id  # ID of the source for this frame

            # object properties
            obj_class_id = obj_meta.class_id
            obj_confidence = obj_meta.confidence
            rect_params = obj_meta.rect_params

            print("Source id: ", source_id)
            print(f"Object Class ID: {obj_class_id} ({class_names[obj_class_id]})")
            print(f"Confidence: {obj_confidence:.2f}")
            print(f"Bounding Box: x={rect_params.left}, y={rect_params.top}, "
                  f"width={rect_params.width}, height={rect_params.height}")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    """
    The function is called when a new pad is created by the decodebin. 
    The function checks if the new pad is for video and not audio. 
    If the new pad is for video, the function checks if the pad caps contain NVMM memory features. 
    If the pad caps contain NVMM memory features, the function links the decodebin pad to the source bin
    ghost pad. 
    If the pad caps do not contain NVMM memory features, the function prints an error message.
    :param decodebin: The decodebin element that is creating the new pad
    :param decoder_src_pad: The source pad created by the decodebin element
    :param data: This is the data that was passed to the callback function. In this case, it is the
    source_bin
    """
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """
    If the child added to the decodebin is another decodebin, connect to its child-added signal. If the
    child added is a source, set its drop-on-latency property to True.
    
    :param child_proxy: The child element that was added to the decodebin
    :param Object: The object that emitted the signal
    :param name: The name of the element that was added
    :param user_data: This is a pointer to the data that you want to pass to the callback function
    """
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    """
    It creates a GstBin, adds a uridecodebin to it, and connects the uridecodebin's pad-added signal to
    a callback function
    
    :param index: The index of the source bin
    :param uri: The URI of the video file to be played
    :return: A bin with a uri decode bin and a ghost pad.
    """
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def make_element(element_name, i):
    """
    Creates a Gstreamer element with unique name
    Unique name is created by adding element type and index e.g. `element_name-i`
    Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
    :param element_name: The name of the element to create
    :param i: the index of the element in the pipeline
    :return: A Gst.Element object
    """
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element


def main(args, requested_pgie=None, config=None, disable_probe=False):
    input_sources = args
    number_sources = len(input_sources)
    global perf_data
    perf_data = PERF_DATA(number_sources)

    platform_info = PlatformInfo()
    Gst.init(None)

    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating streammux \n ")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = input_sources[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pipeline.add(queue1)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvtiler \n ")
    nvtiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not nvtiler:
        sys.stderr.write(" Unable to create nvtiler \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    streammux.set_property("width", 640)
    streammux.set_property("height", 540)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property("config-file-path", "config_infer_primary_yoloV8.txt")

    nvtiler.set_property("rows", int(number_sources**0.5))
    nvtiler.set_property("columns", int((number_sources + int(number_sources**0.5) - 1) // int(number_sources**0.5)))
    nvtiler.set_property("width", 1280)  # Adjust based on your requirements
    nvtiler.set_property("height", 720)  # Adjust based on your requirements

    print("Creating Second Pgie \n")
    second_pgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    if not second_pgie:
        sys.stderr.write(" Unable to create second pgie \n")

    #second_pgie.set_property("config-file-path", "config_infer_secondary_model.txt")
    second_pgie.set_property("config-file-path", "config_infer_primary_yoloV8_lapon.txt")
    second_pgie.set_property("batch-size", number_sources)

    pipeline.add(second_pgie)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(nvtiler)

    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(nvtiler)

    queue = Gst.ElementFactory.make("queue", "queue")
    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvdsosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = (
        Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not platform_info.is_integrated_gpu()
        else Gst.ElementFactory.make("nv3dsink", "nv3d-video-renderer")
    )

    if not queue or not nvvideoconvert or not nvdsosd or not sink:
        sys.stderr.write("Unable to create downstream elements \n")

    pipeline.add(queue)
    pipeline.add(nvvideoconvert)
    pipeline.add(nvdsosd)
    pipeline.add(sink)

    nvtiler.link(queue)
    queue.link(nvvideoconvert)
    nvvideoconvert.link(nvdsosd)
    nvdsosd.link(sink)

    print("Creating nvtracker \n")
    nvtracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not nvtracker:
        sys.stderr.write(" Unable to create nvtracker \n")

    config_tracker = "dstest2_tracker_config.txt"
    nvtracker.set_property("ll-config-file", config_tracker)
    nvtracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")

    pipeline.add(nvtracker)

    streammux.link(pgie)
    pgie.link(nvtracker)
    nvtracker.link(nvvidconv)

    print("Linking elements in the Pipeline \n")
    # create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass
    # Cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(prog="deepstream_demux_multi_in_multi_out.py", 
        description="deepstream-demux-multi-in-multi-out takes multiple URI streams as input" \
            "and uses `nvstreamdemux` to split batches and output separate buffer/streams")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=["a"],
        required=True,
    )

    args = parser.parse_args()
    stream_paths = args.input
    return stream_paths


if __name__ == "__main__":
    stream_paths = parse_args()
    sys.exit(main(stream_paths))
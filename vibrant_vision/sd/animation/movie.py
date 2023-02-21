import subprocess
import os
import numpy as np
from tqdm import tqdm
import cv2
from typing import Callable, List, Optional, Union
import ffmpeg

import logging

logger = logging.getLogger(__name__)


class MovieSaver:
    def __init__(
        self,
        fp_out: str,
        fps: int = 24,
        shape_hw: List[int] = None,
        crf: int = 24,
        codec: str = "libx264",
        preset: str = "fast",
        pix_fmt: str = "yuv420p",
        silent_ffmpeg: bool = True,
    ):
        r"""
        Initializes movie saver class - a human friendly ffmpeg wrapper.
        After you init the class, you can dump numpy arrays x into moviesaver.write_frame(x).
        Don't forget toi finalize movie file with moviesaver.finalize().
        Args:
            fp_out: str
                Output file name. If it already exists, it will be deleted.
            fps: int
                Frames per second.
            shape_hw: List[int, int]
                Output shape, optional argument. Can be initialized automatically when first frame is written.
            crf: int
                ffmpeg doc: the range of the CRF scale is 0–51, where 0 is lossless
                (for 8 bit only, for 10 bit use -qp 0), 23 is the default, and 51 is worst quality possible.
                A lower value generally leads to higher quality, and a subjectively sane range is 17–28.
                Consider 17 or 18 to be visually lossless or nearly so;
                it should look the same or nearly the same as the input but it isn't technically lossless.
                The range is exponential, so increasing the CRF value +6 results in
                roughly half the bitrate / file size, while -6 leads to roughly twice the bitrate.
            codec: int
                Number of diffusion steps. Larger values will take more compute time.
            preset: str
                Choose between ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.
                ffmpeg doc: A preset is a collection of options that will provide a certain encoding speed
                to compression ratio. A slower preset will provide better compression
                (compression is quality per filesize).
                This means that, for example, if you target a certain file size or constant bit rate,
                you will achieve better quality with a slower preset. Similarly, for constant quality encoding,
                you will simply save bitrate by choosing a slower preset.
            pix_fmt: str
                Pixel format. Run 'ffmpeg -pix_fmts' in your shell to see all options.
            silent_ffmpeg: bool
                Surpress the output from ffmpeg.
        """
        if len(os.path.split(fp_out)[0]) > 0:
            assert os.path.isdir(os.path.split(fp_out)[0]), "Directory does not exist!"

        self.fp_out = fp_out
        self.fps = fps
        self.crf = crf
        self.pix_fmt = pix_fmt
        self.codec = codec
        self.preset = preset
        self.silent_ffmpeg = silent_ffmpeg

        if os.path.isfile(fp_out):
            os.remove(fp_out)

        self.init_done = False
        self.nmb_frames = 0
        if shape_hw is None:
            self.shape_hw = [-1, 1]
        else:
            if len(shape_hw) == 2:
                shape_hw.append(3)
            self.shape_hw = shape_hw
            self.initialize()

        logger.info(f"MovieSaver initialized. fps={fps} crf={crf} pix_fmt={pix_fmt} codec={codec} preset={preset}")

    def initialize(self):
        args = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(self.shape_hw[1], self.shape_hw[0]),
                framerate=self.fps,
            )
            .output(self.fp_out, crf=self.crf, pix_fmt=self.pix_fmt, c=self.codec, preset=self.preset)
            .overwrite_output()
            .compile()
        )
        if self.silent_ffmpeg:
            self.ffmpg_process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )
        else:
            self.ffmpg_process = subprocess.Popen(args, stdin=subprocess.PIPE)
        self.init_done = True
        self.shape_hw = tuple(self.shape_hw)
        logger.info(f"Initialization done. Movie shape: {self.shape_hw}")

    def write_frame(self, out_frame: np.ndarray):
        r"""
        Function to dump a numpy array as frame of a movie.
        Args:
            out_frame: np.ndarray
                Numpy array, in np.uint8 format. Convert with np.astype(x, np.uint8).
                Dim 0: y
                Dim 1: x
                Dim 2: RGB
        """

        assert out_frame.dtype == np.uint8, "Convert to np.uint8 before"
        assert len(out_frame.shape) == 3, "out_frame needs to be three dimensional, Y X C"
        assert out_frame.shape[2] == 3, f"need three color channels, but you provided {out_frame.shape[2]}."

        if not self.init_done:
            self.shape_hw = out_frame.shape
            self.initialize()

        assert (
            self.shape_hw == out_frame.shape
        ), f"You cannot change the image size after init. Initialized with {self.shape_hw}, out_frame {out_frame.shape}"

        # write frame
        self.ffmpg_process.stdin.write(out_frame.astype(np.uint8).tobytes())

        self.nmb_frames += 1

    def finalize(self):
        r"""
        Call this function to finalize the movie. If you forget to call it your movie will be garbage.
        """
        if self.nmb_frames == 0:
            logger.warning("You did not write any frames yet! nmb_frames = 0. Cannot save.")
            return
        self.ffmpg_process.stdin.close()
        self.ffmpg_process.wait()
        duration = int(self.nmb_frames / self.fps)
        logger.info(f"Movie saved, {duration}s playtime, watch here: \n{self.fp_out}")


def concatenate_movies(fp_final: str, list_fp_movies: List[str]):
    r"""
    Concatenate multiple movie segments into one long movie, using ffmpeg.
    Parameters
    ----------
    fp_final : str
        Full path of the final movie file. Should end with .mp4
    list_fp_movies : list[str]
        List of full paths of movie segments.
    """
    assert fp_final[-4] == ".", "fp_final seems to miss file extension: {fp_final}"
    for fp in list_fp_movies:
        assert os.path.isfile(fp), f"Input movie does not exist: {fp}"
        assert os.path.getsize(fp) > 100, f"Input movie seems empty: {fp}"

    if os.path.isfile(fp_final):
        os.remove(fp_final)

    # make a list for ffmpeg
    list_concat = []
    for fp_part in list_fp_movies:
        list_concat.append(f"""file '{fp_part}'""")

    # save this list
    fp_list = "tmp_move.txt"
    with open(fp_list, "w") as fa:
        for item in list_concat:
            fa.write("%s\n" % item)

    cmd = f"ffmpeg -f concat -safe 0 -i {fp_list} -c copy {fp_final}"
    dp_movie = os.path.split(fp_final)[0]
    subprocess.call(cmd, shell=True)
    os.remove(fp_list)
    if os.path.isfile(fp_final):
        logger.info(f"concatenate_movies: success! Watch here: {fp_final}")


class MovieReader:
    r"""
    Class to read in a movie.
    """

    def __init__(self, fp_movie):
        self.video_player_object = cv2.VideoCapture(fp_movie)
        self.nmb_frames = int(self.video_player_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_movie = int(self.video_player_object.get(cv2.CAP_PROP_FPS))
        self.shape = [100, 100, 3]
        self.shape_is_set = False

    def get_next_frame(self):
        success, image = self.video_player_object.read()
        if success:
            if not self.shape_is_set:
                self.shape_is_set = True
                self.shape = image.shape
            return image
        else:
            return np.zeros(self.shape)

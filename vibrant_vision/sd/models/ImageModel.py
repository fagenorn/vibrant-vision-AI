import logging
import re
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

from vibrant_vision.sd import constants

logger = logging.getLogger(__name__)
dtype = torch.float16
device = constants.device

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


class BlenderPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, vae, tokenizer, text_encoder, controlnet):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            controlnet=controlnet,
        )

        self.width = 512
        self.height = 512
        self.num_inference_steps = 21
        self.guidance_scale = 8.0
        self.down_sample_factor = 8
        self.channels = self.unet.in_channels

        self.scheduler = UniPCMultistepScheduler.from_config(scheduler.config)

    def __get_samples(self, latents):
        latents = 1 / 0.18215 * latents
        return self.vae.decode(latents).sample

    def __get_images(self, samples):
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = 255 * samples[0, :, :].permute([1, 2, 0]).cpu().numpy()
        return samples.astype("uint8")

    @torch.no_grad()
    def decode_latents(self, latents):
        samples = self.__get_samples(latents)
        image = self.__get_images(samples)

        return image

    @torch.no_grad()
    def get_text_embeds(
        self,
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        max_embeddings_multiples=3,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
            pipe=self,
            prompt=prompt,
            uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
            max_embeddings_multiples=max_embeddings_multiples,
        )
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def controlnet_hint_conversion(self, controlnet_hint):
        channels = 3
        if isinstance(controlnet_hint, torch.Tensor):
            shape_cwh = (channels, self.height, self.width)
            shape_bchw = (1, channels, self.height, self.width)
            shape_nchw = (1, channels, self.height, self.width)
            if controlnet_hint.shape in [shape_cwh, shape_bchw, shape_nchw]:
                controlnet_hint = controlnet_hint.to(dtype=self.controlnet.dtype, device=self.controlnet.device)
                if controlnet_hint.shape != shape_nchw:
                    controlnet_hint = controlnet_hint.repeat(1, 1, 1, 1)
                return controlnet_hint
            else:
                raise ValueError(
                    f"controlnet_hint must be of shape {shape_cwh}, {shape_bchw}, or {shape_nchw}, but is {controlnet_hint.shape}"
                )
        elif isinstance(controlnet_hint, np.ndarray):
            # np.ndarray: acceptable shape is any of hw, hwc, bhwc(b==1) or bhwc(b==num_images_per_promot)
            # hwc is opencv compatible image format. Color channel must be BGR Format.
            if controlnet_hint.shape == (self.height, self.width):
                controlnet_hint = np.repeat(controlnet_hint[:, :, np.newaxis], channels, axis=2)  # hw -> hwc(c==3)
            shape_hwc = (self.height, self.width, channels)
            shape_bhwc = (1, self.height, self.width, channels)
            shape_nhwc = (1, self.height, self.width, channels)
            if controlnet_hint.shape in [shape_hwc, shape_bhwc, shape_nhwc]:
                controlnet_hint = torch.from_numpy(controlnet_hint.copy())
                controlnet_hint = controlnet_hint.to(dtype=self.controlnet.dtype, device=self.controlnet.device)
                controlnet_hint /= 255.0
                if controlnet_hint.shape != shape_nhwc:
                    controlnet_hint = controlnet_hint.repeat(1, 1, 1, 1)
                controlnet_hint = controlnet_hint.permute(0, 3, 1, 2)  # b h w c -> b c h w
                return controlnet_hint
            else:
                raise ValueError(
                    f"Acceptble shape of `controlnet_hint` are any of ({self.width}, {channels}), "
                    + f"({self.height}, {self.width}, {channels}), "
                    + f"(1, {self.height}, {self.width}, {channels}) or "
                    + f"({1}, {channels}, {self.height}, {self.width}) but is {controlnet_hint.shape}"
                )

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds,
        latents,
        list_latents_mixing=None,
        mixing_coeffs=0.0,
        idx_start=0,
        spatial_mask=None,
        return_image=False,
        controlnet_hint=None,
        controlnet_guidance_percent=1.0,
    ):
        if isinstance(mixing_coeffs, list):
            assert len(mixing_coeffs) == self.num_inference_steps
            list_mixing_coeffs = mixing_coeffs
        elif isinstance(mixing_coeffs, float):
            list_mixing_coeffs = [mixing_coeffs] * self.num_inference_steps
        else:
            raise ValueError("mixing_coeffs must be a float or a list of floats")

        if controlnet_hint is not None:
            controlnet_hint = self.controlnet_hint_conversion(controlnet_hint)

        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps

        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = latents.clone()
        list_latents_out = []
        num_warmup_steps = len(timesteps) - self.num_inference_steps * self.scheduler.order
        with self.progress_bar(total=self.num_inference_steps) as progress_bar:
            for i, step in enumerate(timesteps):
                if i < idx_start:
                    list_latents_out.append(None)
                    progress_bar.update()
                    continue
                elif i == idx_start:
                    latents = latents.clone()

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, step)

                if i > 0 and list_mixing_coeffs[i] > 0:
                    latents_mixtarget = list_latents_mixing[i - 1].clone()
                    latents = interpolate_spherical(latents, latents_mixtarget, list_mixing_coeffs[i])

                if spatial_mask is not None and list_latents_mixing is not None:
                    latents = interpolate_spherical(latents, list_latents_mixing[i - 1], 1 - spatial_mask)

                noise_pred = None
                stop_controlnet_guidance = (
                    i - idx_start >= (self.num_inference_steps - idx_start) * controlnet_guidance_percent
                )
                if controlnet_hint is not None and not stop_controlnet_guidance:
                    control = self.controlnet(
                        latent_model_input, step, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint
                    )
                    noise_pred = self.unet(
                        latent_model_input, step, encoder_hidden_states=prompt_embeds, control=control
                    ).sample
                else:
                    noise_pred = self.unet(latent_model_input, step, encoder_hidden_states=prompt_embeds).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, step, latents).prev_sample

                list_latents_out.append(latents.clone())

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if return_image:
            image = self.decode_latents(latents)
            return image
        else:
            return list_latents_out


@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    r"""
    Helper function to correctly mix two random variables using spherical interpolation.
    See https://en.wikipedia.org/wiki/Slerp
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0:
            First tensor for interpolation
        p1:
            Second tensor for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """

    if p0.dtype == torch.float16:
        recast_to = "fp16"
    else:
        recast_to = "fp32"

    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    if recast_to == "fp16":
        interp = interp.half()
    elif recast_to == "fp32":
        interp = interp.float()

    return interp


@torch.no_grad()
def offset_latents(latents, offset_coords_2d):
    """
    Helper function to wrap latents around the image.
    Args:
        latents:
            Latents to wrap
        offset_coords_2d:
            Offset coordinates to use for wrapping
    """
    return torch.nn.functional.grid_sample(
        latents, offset_coords_2d, mode="bicubic", padding_mode="border", align_corners=False
    )


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = pipe.text_encoder(text_input_chunk)[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings = pipe.text_encoder(text_input)[0]
    return text_embeddings


def get_weighted_text_embeddings(
    pipe,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
    Args:
        pipe (`StableDiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [
            token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1] for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None

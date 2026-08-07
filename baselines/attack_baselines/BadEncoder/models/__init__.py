from .clip_model import CLIP


def get_encoder_architecture_usage(args):
    if args.encoder_usage_info != 'CLIP':
        raise ValueError('Only CLIP is supported by this reproduction.')
    return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)

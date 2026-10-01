"""Explicit local-video tracking. Import and --help perform no service setup."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import math
import os
from pathlib import Path
import sys

from video_pipeline import process_video


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Expected a positive finite number')
    return number


def arguments(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--weights', type=Path, required=True, help='Trusted local Ultralytics .pt weights')
    cli.add_argument('--encoder', type=Path, required=True, help='Trusted local DeepSORT frozen .pb graph')
    cli.add_argument('--fps', type=positive, help='Override missing/invalid constant frame rate')
    cli.add_argument('--confidence', type=float, default=0.5)
    cli.add_argument('--alert-after', type=positive, default=3.0)
    cli.add_argument('--telegram', action='store_true', help='Opt in to uploading frames to Telegram')
    args = cli.parse_args(argv)
    for path in (args.input, args.weights, args.encoder):
        if not path.is_file():
            cli.error('Input video and both model files must exist locally')
    if args.weights.suffix.lower() != '.pt' or args.encoder.suffix.lower() != '.pb':
        cli.error('Expected local .pt detector weights and a .pb encoder')
    if args.output.exists() or args.output.is_symlink():
        cli.error('Output already exists; select a new file')
    if args.output.suffix.lower() not in ('.avi', '.mp4') or not args.output.parent.is_dir():
        cli.error('Output needs an existing parent directory and .avi or .mp4 suffix')
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        cli.error('Confidence must be between 0 and 1')
    if args.telegram and (not os.getenv('TOKEN') or not os.getenv('CHAT_ID')):
        cli.error('Telegram opt-in requires TOKEN and CHAT_ID environment variables')
    return args


def telegram_sender(bot, chat_id):
    async def send(frame, track_id):
        import cv2
        from aiogram.types import BufferedInputFile

        ok, encoded = cv2.imencode('.jpg', frame)
        if not ok:
            raise ValueError('JPEG encoding failed')
        photo = BufferedInputFile(encoded.tobytes(), filename='tracking-alert.jpg')
        await bot.send_photo(chat_id=chat_id, photo=photo, caption=f'Track {track_id}', request_timeout=20)
    return send


async def run(args):
    from ultralytics import YOLO
    from deepsort_tracker import Tracker

    async with AsyncExitStack() as resources:
        tracker = Tracker(args.encoder.resolve())
        resources.callback(tracker.close)
        detector = YOLO(str(args.weights.resolve()))
        if getattr(detector, 'names', {}).get(0) != 'person':
            raise ValueError('Expected COCO-compatible detector with person at class 0')
        alert = None
        if args.telegram:
            from aiogram import Bot
            bot = Bot(token=os.environ['TOKEN'])
            resources.push_async_callback(bot.session.close)
            alert = telegram_sender(bot, os.environ['CHAT_ID'])
        return await process_video(args.input, args.output, detector, tracker,
                                   confidence=args.confidence, fps_override=args.fps,
                                   alert_after=args.alert_after, send_alert=alert)


def main(argv=None):
    args = arguments(argv)
    try:
        result = asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        # Provider exception strings may include token-bearing request URLs.
        print('Run failed: ' + type(error).__name__, file=sys.stderr)
        return 1
    print(f"Processed {result['frames']} frames; sent {result['alerts']} alerts")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

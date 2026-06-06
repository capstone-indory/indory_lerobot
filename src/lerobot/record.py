#!/usr/bin/env python

"""Compatibility entrypoint for XLeRobot documentation examples.

XLeRobot's setup guide calls this module directly. Keep the path available while
delegating to the maintained LeRobot recording script in this checkout.
"""

from lerobot.scripts.lerobot_record import main


if __name__ == "__main__":
    main()

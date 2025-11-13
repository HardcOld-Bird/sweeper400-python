# pyright: basic
from sweeper400.move import (  # type: ignore
    MotorController,
)

mc = MotorController()
# mc.get_hardware_info()

mc.get_current_position_2D()

mc.move_absolute_1D(0, 80.0)
mc.move_absolute_2D(160.0, 160.0)

mc.calibrate_all_axis()

mc.get_axis_status(1)

del mc

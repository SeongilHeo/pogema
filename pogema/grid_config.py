import sys
from typing import Optional, Union
from pydantic import field_validator, model_validator, ValidationInfo, ConfigDict

from pogema.utils import CommonSettings

from typing_extensions import Literal


class GridConfig(CommonSettings, ):
    model_config = ConfigDict(validate_default=True)
    on_target: Literal['finish', 'nothing', 'restart'] = 'finish'
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size: int = 8
    density: float = 0.3
    obs_radius: int = 5
    agents_xy: Optional[list] = None
    targets_xy: Optional[list] = None
    num_agents: Optional[int] = None
    possible_agents_xy: Optional[list] = None
    possible_targets_xy: Optional[list] = None
    collision_system: Literal['block_both', 'priority', 'soft'] = 'priority'
    persistent: bool = False
    observation_type: Literal['POMAPF', 'MAPF', 'default'] = 'default'
    map: Optional[Union[list, str]] = None

    map_name: Optional[str] = None

    integration: Optional[Literal['SampleFactory', 'PyMARL', 'rllib', 'gymnasium', 'PettingZoo']] = None
    max_episode_steps: int = 64
    auto_reset: Optional[bool] = None

    @model_validator(mode='before')
    @classmethod
    def _preprocess_map_and_dimensions(cls, values):
        if not isinstance(values, dict):
            return values

        # Step 1: Process string map
        map_val = values.get('map')
        if isinstance(map_val, str):
            free = values.get('FREE', 0)
            obstacle = values.get('OBSTACLE', 1)
            v, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy = cls.str_map_to_list(
                map_val, free, obstacle
            )
            if agents_xy and targets_xy and values.get('agents_xy') is not None and values.get(
                    'targets_xy') is not None:
                raise KeyError("""Can't create task. Please provide agents_xy and targets_xy only once.
                Either with parameters or with a map.""")
            if (agents_xy or targets_xy) and (possible_agents_xy or possible_targets_xy):
                raise KeyError("""Can't create task. Mark either possible locations or precise ones.""")
            elif agents_xy and targets_xy:
                values['agents_xy'] = agents_xy
                values['targets_xy'] = targets_xy
                values['num_agents'] = len(agents_xy)
            elif (values.get('agents_xy') is None or values.get(
                    'targets_xy') is None) and possible_agents_xy and possible_targets_xy:
                values['possible_agents_xy'] = possible_agents_xy
                values['possible_targets_xy'] = possible_targets_xy
            values['map'] = v

        # Step 2: Extract dimensions from map list
        map_val = values.get('map')
        if isinstance(map_val, list):
            height = len(map_val)
            width = 0
            area = 0
            for line in map_val:
                width = max(width, len(line))
                area += len(line)
            values['size'] = max(width, height)
            values['width'] = width
            values['height'] = height
            if area > 0:
                values['density'] = sum(sum(line) for line in map_val) / area

        # Step 3: Validate/set dimensions
        width_provided = values.get('width') is not None
        height_provided = values.get('height') is not None

        if width_provided and not height_provided:
            raise ValueError("Invalid dimension configuration. Please provide height.")
        elif not width_provided and height_provided:
            raise ValueError("Invalid dimension configuration. Please provide width.")

        if not width_provided and not height_provided:
            values['width'] = values.get('size', 8)
            values['height'] = values.get('size', 8)

        w = values.get('width') or 0
        h = values.get('height') or 0
        if values.get('size') is None or values.get('size') != max(w, h):
            values['size'] = max(w, h)

        # Step 4: Validate agent/target positions within bounds
        width = values.get('width')
        height = values.get('height')

        if width is not None and height is not None:
            agents_xy = values.get('agents_xy')
            if agents_xy is not None:
                cls.check_positions(agents_xy, width, height)

            targets_xy = values.get('targets_xy')
            if targets_xy is not None and len(targets_xy) > 0:
                first_element = targets_xy[0]
                if isinstance(first_element[0], (list, tuple)):
                    for agent_goals in targets_xy:
                        cls.check_positions(agent_goals, width, height)
                else:
                    cls.check_positions(targets_xy, width, height)

        return values

    @field_validator('seed')
    @classmethod
    def seed_initialization(cls, v):
        assert v is None or (0 <= v < sys.maxsize), "seed must be in [0, " + str(sys.maxsize) + ']'
        return v

    @staticmethod
    def _validate_dimension(v, field_name):
        if v is not None:
            if field_name == 'size':
                assert 2 <= v <= 4096, f"{field_name} must be in [2, 4096]"
            else:
                assert 1 <= v <= 4096, f"{field_name} must be in [1, 4096]"
        return v

    @field_validator('size')
    @classmethod
    def size_restrictions(cls, v):
        return cls._validate_dimension(v, 'size')

    @field_validator('width')
    @classmethod
    def width_restrictions(cls, v):
        return cls._validate_dimension(v, 'width')

    @field_validator('height')
    @classmethod
    def height_restrictions(cls, v):
        return cls._validate_dimension(v, 'height')

    @field_validator('density')
    @classmethod
    def density_restrictions(cls, v):
        assert 0.0 <= v <= 1, "density must be in [0, 1]"
        return v

    @field_validator('agents_xy')
    @classmethod
    def agents_xy_validation(cls, v):
        if v is not None:
            if not isinstance(v, (list, tuple)):
                raise ValueError("agents_xy must be a list")
            for position in v:
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    raise ValueError("Position must be a list/tuple of length 2")
                if not all(isinstance(coord, int) for coord in position):
                    raise ValueError("Position coordinates must be integers")
        return v

    @field_validator('targets_xy')
    @classmethod
    def targets_xy_validation(cls, v, info: ValidationInfo):
        if v is not None:
            if not v or not isinstance(v, (list, tuple)):
                raise ValueError("targets_xy must be a list")

            first_element = v[0]
            if not isinstance(first_element, (list, tuple)):
                raise ValueError("Invalid targets_xy format")

            if isinstance(first_element[0], (list, tuple)):
                for agent_goals in v:
                    if not isinstance(agent_goals, (list, tuple)) or len(agent_goals) < 2:
                        raise ValueError("Each agent must have at least two goals in the sequence")
                    for position in agent_goals:
                        if not isinstance(position, (list, tuple)) or len(position) != 2:
                            raise ValueError("Position must be a list/tuple of length 2")
                        if not all(isinstance(coord, int) for coord in position):
                            raise ValueError("Position coordinates must be integers")
            else:
                on_target = info.data.get('on_target', 'finish')
                if on_target == 'restart':
                    raise ValueError("on_target='restart' requires goal sequences, not single goals. Use format: targets_xy: [[[x1,y1],[x2,y2]], [[x3,y3],[x4,y4]]]")
                for position in v:
                    if not isinstance(position, (list, tuple)) or len(position) != 2:
                        raise ValueError("Position must be a list/tuple of length 2")
                    if not all(isinstance(coord, int) for coord in position):
                        raise ValueError("Position coordinates must be integers")
        return v

    @staticmethod
    def check_positions(v, width, height):
        for position in v:
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError("Position must be a list/tuple of length 2")
            x, y = position
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("Position coordinates must be integers")
            if not (0 <= x < height and 0 <= y < width):
                raise IndexError(f"Position is out of bounds! {position} is not in [{0}, {height}] x [{0}, {width}]")

    @field_validator('num_agents')
    @classmethod
    def num_agents_must_be_positive(cls, v, info: ValidationInfo):
        if v is None:
            agents_xy = info.data.get('agents_xy')
            if agents_xy:
                v = len(agents_xy)
            else:
                v = 1
        assert 1 <= v <= 10000000, "num_agents must be in [1, 10000000]"
        return v

    @field_validator('obs_radius')
    @classmethod
    def obs_radius_must_be_positive(cls, v):
        assert 1 <= v <= 128, "obs_radius must be in [1, 128]"
        return v

    @staticmethod
    def str_map_to_list(str_map, free, obstacle):
        obstacles = []
        agents = {}
        targets = {}
        possible_agents_xy = []
        possible_targets_xy = []
        special_chars = {'@', '$', '!'}

        for row_idx, line in enumerate(str_map.split()):
            row = []
            for col_idx, char in enumerate(line):
                position = (row_idx, col_idx)

                if char == '.':
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif char == '#':
                    row.append(obstacle)
                elif char in special_chars:
                    row.append(free)
                    if char == '@':
                        possible_agents_xy.append(position)
                    elif char == '$':
                        possible_targets_xy.append(position)
                elif 'A' <= char <= 'Z':
                    targets[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif 'a' <= char <= 'z':
                    agents[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {row_idx}")

            if row:
                assert len(obstacles[-1]) == len(row) if obstacles else True, f"Wrong string size for row {row_idx};"
                obstacles.append(row)

        agents_xy = [[x, y] for _, (x, y) in sorted(agents.items())]
        targets_xy = [[x, y] for _, (x, y) in sorted(targets.items())]

        assert len(targets_xy) == len(agents_xy), "Mismatch in number of agents and targets."

        if not any(char in special_chars for char in str_map):
            possible_agents_xy, possible_targets_xy = None, None

        return obstacles, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy

    def update_config(self, **kwargs):
        current_values = self.model_dump()

        if 'size' in kwargs:
            current_values.pop('width', None)
            current_values.pop('height', None)
        elif 'width' in kwargs or 'height' in kwargs:
            current_values.pop('size', None)
        current_values.update(kwargs)
        new_instance = GridConfig(**current_values)

        for field_name in self.model_fields:
            setattr(self, field_name, getattr(new_instance, field_name))

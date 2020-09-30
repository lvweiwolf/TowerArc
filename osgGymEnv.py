# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _osgGymEnv
else:
    import _osgGymEnv

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _osgGymEnv.delete_SwigPyIterator

    def value(self):
        return _osgGymEnv.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _osgGymEnv.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _osgGymEnv.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _osgGymEnv.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _osgGymEnv.SwigPyIterator_equal(self, x)

    def copy(self):
        return _osgGymEnv.SwigPyIterator_copy(self)

    def next(self):
        return _osgGymEnv.SwigPyIterator_next(self)

    def __next__(self):
        return _osgGymEnv.SwigPyIterator___next__(self)

    def previous(self):
        return _osgGymEnv.SwigPyIterator_previous(self)

    def advance(self, n):
        return _osgGymEnv.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _osgGymEnv.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _osgGymEnv.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _osgGymEnv.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _osgGymEnv.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _osgGymEnv.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _osgGymEnv.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _osgGymEnv:
_osgGymEnv.SwigPyIterator_swigregister(SwigPyIterator)

class PointVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _osgGymEnv.PointVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _osgGymEnv.PointVector___nonzero__(self)

    def __bool__(self):
        return _osgGymEnv.PointVector___bool__(self)

    def __len__(self):
        return _osgGymEnv.PointVector___len__(self)

    def __getslice__(self, i, j):
        return _osgGymEnv.PointVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _osgGymEnv.PointVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _osgGymEnv.PointVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _osgGymEnv.PointVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _osgGymEnv.PointVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _osgGymEnv.PointVector___setitem__(self, *args)

    def pop(self):
        return _osgGymEnv.PointVector_pop(self)

    def append(self, x):
        return _osgGymEnv.PointVector_append(self, x)

    def empty(self):
        return _osgGymEnv.PointVector_empty(self)

    def size(self):
        return _osgGymEnv.PointVector_size(self)

    def swap(self, v):
        return _osgGymEnv.PointVector_swap(self, v)

    def begin(self):
        return _osgGymEnv.PointVector_begin(self)

    def end(self):
        return _osgGymEnv.PointVector_end(self)

    def rbegin(self):
        return _osgGymEnv.PointVector_rbegin(self)

    def rend(self):
        return _osgGymEnv.PointVector_rend(self)

    def clear(self):
        return _osgGymEnv.PointVector_clear(self)

    def get_allocator(self):
        return _osgGymEnv.PointVector_get_allocator(self)

    def pop_back(self):
        return _osgGymEnv.PointVector_pop_back(self)

    def erase(self, *args):
        return _osgGymEnv.PointVector_erase(self, *args)

    def __init__(self, *args):
        _osgGymEnv.PointVector_swiginit(self, _osgGymEnv.new_PointVector(*args))

    def push_back(self, x):
        return _osgGymEnv.PointVector_push_back(self, x)

    def front(self):
        return _osgGymEnv.PointVector_front(self)

    def back(self):
        return _osgGymEnv.PointVector_back(self)

    def assign(self, n, x):
        return _osgGymEnv.PointVector_assign(self, n, x)

    def resize(self, *args):
        return _osgGymEnv.PointVector_resize(self, *args)

    def insert(self, *args):
        return _osgGymEnv.PointVector_insert(self, *args)

    def reserve(self, n):
        return _osgGymEnv.PointVector_reserve(self, n)

    def capacity(self):
        return _osgGymEnv.PointVector_capacity(self)
    __swig_destroy__ = _osgGymEnv.delete_PointVector

# Register PointVector in _osgGymEnv:
_osgGymEnv.PointVector_swigregister(PointVector)

class Point3D(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _osgGymEnv.Point3D_swiginit(self, _osgGymEnv.new_Point3D(*args))
    x = property(_osgGymEnv.Point3D_x_get, _osgGymEnv.Point3D_x_set)
    y = property(_osgGymEnv.Point3D_y_get, _osgGymEnv.Point3D_y_set)
    z = property(_osgGymEnv.Point3D_z_get, _osgGymEnv.Point3D_z_set)
    __swig_destroy__ = _osgGymEnv.delete_Point3D

# Register Point3D in _osgGymEnv:
_osgGymEnv.Point3D_swigregister(Point3D)

def Point3DCopy(rhs):
    val = _osgGymEnv.new_Point3DCopy(rhs)
    return val

class Extents(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _osgGymEnv.Extents_swiginit(self, _osgGymEnv.new_Extents(*args))

    def Intersects(self, minX, minY, minZ, maxX, maxY, maxZ):
        return _osgGymEnv.Extents_Intersects(self, minX, minY, minZ, maxX, maxY, maxZ)

    def GetCenter(self):
        return _osgGymEnv.Extents_GetCenter(self)
    min_x = property(_osgGymEnv.Extents_min_x_get, _osgGymEnv.Extents_min_x_set)
    min_y = property(_osgGymEnv.Extents_min_y_get, _osgGymEnv.Extents_min_y_set)
    min_z = property(_osgGymEnv.Extents_min_z_get, _osgGymEnv.Extents_min_z_set)
    max_x = property(_osgGymEnv.Extents_max_x_get, _osgGymEnv.Extents_max_x_set)
    max_y = property(_osgGymEnv.Extents_max_y_get, _osgGymEnv.Extents_max_y_set)
    max_z = property(_osgGymEnv.Extents_max_z_get, _osgGymEnv.Extents_max_z_set)
    __swig_destroy__ = _osgGymEnv.delete_Extents

# Register Extents in _osgGymEnv:
_osgGymEnv.Extents_swigregister(Extents)

def ExtentsCopy(extents):
    val = _osgGymEnv.new_ExtentsCopy(extents)
    return val

class Body(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _osgGymEnv.Body_swiginit(self, _osgGymEnv.new_Body(*args))

    def IsValid(self):
        return _osgGymEnv.Body_IsValid(self)

    def SetId(self, identiy):
        return _osgGymEnv.Body_SetId(self, identiy)

    def GetId(self):
        return _osgGymEnv.Body_GetId(self)

    def SetNode(self, nodePtr):
        return _osgGymEnv.Body_SetNode(self, nodePtr)

    def GetNode(self):
        return _osgGymEnv.Body_GetNode(self)
    __swig_destroy__ = _osgGymEnv.delete_Body

# Register Body in _osgGymEnv:
_osgGymEnv.Body_swigregister(Body)

class TerrainBody(Body):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _osgGymEnv.TerrainBody_swiginit(self, _osgGymEnv.new_TerrainBody(*args))
    extent = property(_osgGymEnv.TerrainBody_extent_get, _osgGymEnv.TerrainBody_extent_set)
    dem = property(_osgGymEnv.TerrainBody_dem_get, _osgGymEnv.TerrainBody_dem_set)
    dom = property(_osgGymEnv.TerrainBody_dom_get, _osgGymEnv.TerrainBody_dom_set)
    __swig_destroy__ = _osgGymEnv.delete_TerrainBody

# Register TerrainBody in _osgGymEnv:
_osgGymEnv.TerrainBody_swigregister(TerrainBody)

class TowerBody(Body):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _osgGymEnv.TowerBody_swiginit(self, _osgGymEnv.new_TowerBody(*args))

    def SetName(self, towerName):
        return _osgGymEnv.TowerBody_SetName(self, towerName)

    def GetName(self):
        return _osgGymEnv.TowerBody_GetName(self)

    def SetModelPath(self, model_file):
        return _osgGymEnv.TowerBody_SetModelPath(self, model_file)

    def GetModelPath(self):
        return _osgGymEnv.TowerBody_GetModelPath(self)

    def SetNode(self, nodePtr):
        return _osgGymEnv.TowerBody_SetNode(self, nodePtr)

    def GetNode(self):
        return _osgGymEnv.TowerBody_GetNode(self)

    def DistanceTo(self, other):
        return _osgGymEnv.TowerBody_DistanceTo(self, other)
    x = property(_osgGymEnv.TowerBody_x_get, _osgGymEnv.TowerBody_x_set)
    y = property(_osgGymEnv.TowerBody_y_get, _osgGymEnv.TowerBody_y_set)
    height = property(_osgGymEnv.TowerBody_height_get, _osgGymEnv.TowerBody_height_set)
    actualHeight = property(_osgGymEnv.TowerBody_actualHeight_get, _osgGymEnv.TowerBody_actualHeight_set)
    name = property(_osgGymEnv.TowerBody_name_get, _osgGymEnv.TowerBody_name_set)
    model_path = property(_osgGymEnv.TowerBody_model_path_get, _osgGymEnv.TowerBody_model_path_set)
    __swig_destroy__ = _osgGymEnv.delete_TowerBody

# Register TowerBody in _osgGymEnv:
_osgGymEnv.TowerBody_swigregister(TowerBody)

def TowerBodyCopy(rhs):
    val = _osgGymEnv.new_TowerBodyCopy(rhs)
    return val

class ArclineBody(Body):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _osgGymEnv.ArclineBody_swiginit(self, _osgGymEnv.new_ArclineBody())
    startTower = property(_osgGymEnv.ArclineBody_startTower_get, _osgGymEnv.ArclineBody_startTower_set)
    endTower = property(_osgGymEnv.ArclineBody_endTower_get, _osgGymEnv.ArclineBody_endTower_set)
    pntsInSpan = property(_osgGymEnv.ArclineBody_pntsInSpan_get, _osgGymEnv.ArclineBody_pntsInSpan_set)
    pntsOutSpan = property(_osgGymEnv.ArclineBody_pntsOutSpan_get, _osgGymEnv.ArclineBody_pntsOutSpan_set)
    lowestPnt = property(_osgGymEnv.ArclineBody_lowestPnt_get, _osgGymEnv.ArclineBody_lowestPnt_set)
    K = property(_osgGymEnv.ArclineBody_K_get, _osgGymEnv.ArclineBody_K_set)
    step = property(_osgGymEnv.ArclineBody_step_get, _osgGymEnv.ArclineBody_step_set)
    __swig_destroy__ = _osgGymEnv.delete_ArclineBody

# Register ArclineBody in _osgGymEnv:
_osgGymEnv.ArclineBody_swigregister(ArclineBody)

def ArclineBodyCopy(rhs):
    val = _osgGymEnv.new_ArclineBodyCopy(rhs)
    return val

class LineBody(Body):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _osgGymEnv.LineBody_swiginit(self, _osgGymEnv.new_LineBody(*args))
    startPnt = property(_osgGymEnv.LineBody_startPnt_get, _osgGymEnv.LineBody_startPnt_set)
    endPnt = property(_osgGymEnv.LineBody_endPnt_get, _osgGymEnv.LineBody_endPnt_set)
    __swig_destroy__ = _osgGymEnv.delete_LineBody

# Register LineBody in _osgGymEnv:
_osgGymEnv.LineBody_swigregister(LineBody)

class World(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _osgGymEnv.World_swiginit(self, _osgGymEnv.new_World())
    __swig_destroy__ = _osgGymEnv.delete_World

    def GetViewer(self):
        return _osgGymEnv.World_GetViewer(self)

    def Step(self, timeStep, velocityIterations, positionIterations):
        return _osgGymEnv.World_Step(self, timeStep, velocityIterations, positionIterations)

    def CreateTerrainBody(self, dem, dom):
        return _osgGymEnv.World_CreateTerrainBody(self, dem, dom)

    def DeleteTerrainBody(self, terrain):
        return _osgGymEnv.World_DeleteTerrainBody(self, terrain)

    def CreateTowerBody(self, x, y, height, model_path):
        return _osgGymEnv.World_CreateTowerBody(self, x, y, height, model_path)

    def DeleteTowerBody(self, tower):
        return _osgGymEnv.World_DeleteTowerBody(self, tower)

    def CreateArclineBody(self, startTower, endTower, K, step):
        return _osgGymEnv.World_CreateArclineBody(self, startTower, endTower, K, step)

    def DeleteArclineBody(self, arcline):
        return _osgGymEnv.World_DeleteArclineBody(self, arcline)

    def CreateLineBody(self, startPnt, endPnt):
        return _osgGymEnv.World_CreateLineBody(self, startPnt, endPnt)

    def DeleteLineBody(self, line):
        return _osgGymEnv.World_DeleteLineBody(self, line)

    def UpdateArcline(self, arcline):
        return _osgGymEnv.World_UpdateArcline(self, arcline)

    def CalcLowestDistance(self, arcline, lowestPnt):
        return _osgGymEnv.World_CalcLowestDistance(self, arcline, lowestPnt)

# Register World in _osgGymEnv:
_osgGymEnv.World_swigregister(World)

class Viewer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, model):
        _osgGymEnv.Viewer_swiginit(self, _osgGymEnv.new_Viewer(model))
    __swig_destroy__ = _osgGymEnv.delete_Viewer

    def StartRender(self):
        return _osgGymEnv.Viewer_StartRender(self)

    def RecordTower(self, tower):
        return _osgGymEnv.Viewer_RecordTower(self, tower)

    def RecordArcline(self, arcline):
        return _osgGymEnv.Viewer_RecordArcline(self, arcline)

    def RecordLine(self, line):
        return _osgGymEnv.Viewer_RecordLine(self, line)

    def DrawRecords(self):
        return _osgGymEnv.Viewer_DrawRecords(self)

# Register Viewer in _osgGymEnv:
_osgGymEnv.Viewer_swigregister(Viewer)




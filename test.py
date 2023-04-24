from numba import njit, types
from numba.experimental import jitclass
import numba

@jitclass
class TestClass:
    a: int

    def __init__(self, a: int) -> None:
        self.a = a

    def retnone(self, x: int) -> int | None:
        return x if x < 10 else None

    def isnone(self, x: int) -> bool:
        return self.retnone(x) is None

@njit
def retnone(x: int) -> int | None:
    return x if x < 10 else None

@njit
def test(x: int) -> None:
    print(retnone(4) is None)
    return None

tst = TestClass(10)
print(tst.retnone(123))
# Быстрый старт

Нужен Python 3.12.

Пакет: [pypi.org/project/topologysegmentation](https://pypi.org/project/topologysegmentation).

```bash
pip install topologysegmentation
pip install "topologysegmentation[ph]"
```

Первая команда ставит Dice, clDice, T-Loss, `pooling` и `critical`. Вторая добавляет `gudhi`, `pot` и `torch-topological` для `hu`, `bm`, `sat` и чисел Бетти.

```python
from topologysegmentation import CLDiceLoss, DiceLoss, SumLoss, loss_shapes

# Кадр (N, 1, H, W) или объём (N, 1, D, H, W). Сигмоида считается внутри лосса.
loss = SumLoss((DiceLoss(), 1.0), (CLDiceLoss(), 0.1))(logits, target)
loss_shapes()  # у каждого имени набор размерностей: 2 — кадр, 3 — объём
```

`dice`, `bce`, `focal`, `combined`, `cldice`, `tloss`, `pooling`, `critical`, `hu` и `bm3d` принимают кадр и объём. `sat` и `bm` только кадр: на объёме они бросают `ValueError`, для объёма есть `bm3d`. Тензор `(D, H, W)` без канала читается как батч плоских кадров.

`betti_numbers` на кадре возвращает `(β0, β1)`, на объёме `(β0, β1, β2)`. При первом вызове, если найден `clang`, `cc` или `gcc`, собирается `cubical_betti.c` из колеса. Без компилятора считается запасной путь на NumPy. `TOPOLOGYSEGMENTATION_NATIVE=0` оставляет только его. `bm` так же собирает `cubical_matching.c`; `TOPOLOGYSEGMENTATION_MATCHING=0` оставляет для него питоновские циклы.

`bm3d` через pip не ставится. Расширение `betti_matching` собирается из [Betti-Matching-3D](https://github.com/nstucki/Betti-Matching-3D), каталог `build` добавляется в `PYTHONPATH`.

Классы и смысл каждого параметра — в [docs/source/losses.rst](docs/source/losses.rst). Краткий список весов и лицензии — в [README.md](README.md).

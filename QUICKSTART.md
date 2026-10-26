# Быстрый старт

Две разные установки. Колесо с PyPI — это лоссы. Команды `train.py` есть только в клоне репозитория.

## Пакет

Нужен Python 3.12.

Пакет: [pypi.org/project/topologysegmentation](https://pypi.org/project/topologysegmentation).

```bash
pip install topologysegmentation
pip install "topologysegmentation[ph]"
```

Первая команда ставит Dice, clDice, T-Loss, `pooling` и `critical`. Вторая добавляет `gudhi`, `pot` и `torch-topological` для `hu`, `bm`, `sat` и чисел Бетти.

```python
from topologysegmentation import DiceLoss, create_loss, loss_shapes

# Кадр (N, 1, H, W) или объём (N, 1, D, H, W). Сигмоида считается внутри лосса.
loss = DiceLoss()(logits, target) + 0.1 * create_loss("cldice")(logits, target)
loss_shapes()  # у каждого имени набор размерностей: 2 — кадр, 3 — объём
```

`dice`, `bce`, `focal`, `combined`, `cldice`, `tloss`, `pooling`, `critical`, `hu` и `bm3d` принимают кадр и объём. `sat` и `bm` только кадр: на объёме они бросают `ValueError`, для объёма есть `bm3d`. Тензор `(D, H, W)` без канала читается как батч плоских кадров.

`betti_numbers` на кадре возвращает `(β0, β1)`, на объёме `(β0, β1, β2)`. При первом вызове, если найден `clang`, `cc` или `gcc`, собирается `cubical_betti.c` из колеса. Без компилятора считается запасной путь на NumPy. `TOPOLOGYSEGMENTATION_NATIVE=0` оставляет только его. `bm` так же собирает `cubical_matching.c`; `TOPOLOGYSEGMENTATION_MATCHING=0` оставляет для него питоновские циклы.

Скорость на текущей машине, вместе с backward:

```bash
python tests/bench_losses.py
```

`bm3d` этой командой не ставится. Расширение `betti_matching` собирается из [Betti-Matching-3D](https://github.com/nstucki/Betti-Matching-3D), каталог `build` добавляется в `PYTHONPATH`.

Полный список лоссов, весов и лицензий — в [README.md](README.md).

## Клон репозитория

`uv sync` ставит пакет в окружение `.venv` и группу `dev` из `pyproject.toml`: трансформер, чтение кадров, тензорборд. Эти пакеты в колесо не входят.

```bash
uv sync
python tests/test_contract.py
python train.py --config configs/experiments/dice.yaml --max-steps 1 --image-size 64 --batch-size 2 --epochs 1
```

Полный прогон:

```bash
python train.py --config configs/experiments/dice.yaml
```

Сравнение конфигов из списка `DEFAULT_CONFIGS` в `experiments/sweep.py`:

```bash
python train.py sweep --max-steps 1 --image-size 64 --batch-size 2
```

Согласие двух аннотаторов на общих кадрах:

```bash
python train.py agreement
```

Кадры VessMAP лежат в `src/VessMAP` и в git не входят. Сплит с сидом 42 — `splits/vessmap_seed42.json`: 80 обучающих, 12 валидационных, 8 тестовых. Каталог прогона `runs/<время>_<имя>/` содержит копию конфига, сплит, `history.json` и `checkpoints/best.pt`. Поле `parts` в истории — уже взвешенный вклад каждого лосса.

Конфиги новых лоссов: `configs/experiments/dice_hu.yaml`, `dice_pooling.yaml`, `dice_critical.yaml`. Веса там уже подобраны под масштаб лосса, см. таблицу в README.

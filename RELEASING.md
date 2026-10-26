# Выкладка на PyPI

Колесо содержит только пакет `topologysegmentation`. Обучение VessMAP, тесты, датасет и сборка `betti_matching` остаются в git.

Имя на PyPI — `topologysegmentation`, импорт тот же: `import topologysegmentation`. Имя `topologylosses` не использовать: на PyPI уже есть проект `topolosses`.

## Версия

Номер записан одной строкой в `src/topologysegmentation/VERSION`. Его читают и пакет, и сборка. Первая публикация — `0.1.0`. Повторно залить ту же версию нельзя: перед следующей сборкой строку нужно увеличить.

## Сборка

Из корня репозитория, Python 3.12:

```bash
uv build
uvx twine check dist/*
```

Появляются `dist/topologysegmentation-0.1.0.tar.gz` и `dist/topologysegmentation-0.1.0-py3-none-any.whl`. Каталог `dist/` в git не входит.

В архиве исходников и в колесе должны быть `LICENSE`, `NOTICE`, `src/topologysegmentation/losses/vendor/tloss/LICENSE`, `src/topologysegmentation/_native/cubical_betti.c` и `src/topologysegmentation/_native/cubical_matching.c`. Не должно быть `experiments/`, `configs/`, `tests/`, `runs/`, `VessMAP`, `Betti-Matching-3D`, собранных библиотек `cubical_betti_*.dylib` и `cubical_matching_*.dylib` и `losses/vendor/gd`.

Колесо остаётся `py3-none-any`: компилятор для `pip install` не нужен. `cubical_betti.c` собирается при первом вызове `betti_numbers`, `cubical_matching.c` — при первом вызове `bm`, если в системе есть `clang`, `cc` или `gcc`. Без компилятора работают запасные пути на Python.

```bash
tar -tzf dist/topologysegmentation-0.1.0.tar.gz
unzip -l dist/topologysegmentation-0.1.0-py3-none-any.whl
```

## Публикация

Нужен токен со страницы https://pypi.org/manage/account/token/ с областью на весь аккаунт или на проект `topologysegmentation`. Токен TestPyPI на основной индекс не действует.

```bash
uvx twine upload dist/*
```

Проверка во временном окружении:

```bash
uv venv /tmp/tl-smoke --python 3.12
uv pip install --python /tmp/tl-smoke/bin/python topologysegmentation
/tmp/tl-smoke/bin/python -c "import topologysegmentation; print(topologysegmentation.__version__, topologysegmentation.available_losses())"
```

`uv venv` не кладёт в окружение исполняемый `pip`, поэтому ставить пакеты нужно через `uv pip --python`.

Страница проекта берёт текст из `README.md`. `pip install topologysegmentation` не ставит `bm3d`: расширение собирается отдельно, лицензия MIT указана в `NOTICE`.

### Инструкция по сборке модели для разметки пробирок на фотографиях

1. Потребуется создать директорию `/images` и перенести туда фотографии

```sh
mkdir images
```

2. Сделать сборку модели

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```sh
cmake --build build 
```

*Должна появиться директория `/build` с собранной моделью

3. Запустить проект

```sh
./build/tube_detector \
    --input   ./images \
    --output  ./images_marked \
    --json    labels.json \
    --config  config.yaml 
```

После успешной сборки и запуска модели, появятся файлы `detect.log` (тут будет лог всех изображений с количеством разметок и статусом наличия пробирок), `labels.json`
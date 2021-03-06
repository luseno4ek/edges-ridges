## Детектирование границ и хребтов изображения

Код можно посмотреть [здесь](https://github.com/luseno4ek/edges-ridges/blob/67f74e73df454a5a320fd9369045cbe6a698813e/mmip_task3.py).

### Описание работы алгоритма

Реализован алгоритм детектирования границ Canny Edge Detector, а также метод выделения хребтовых структур, описанный в [презентации](https://github.com/luseno4ek/edges-ridges/blob/67f74e73df454a5a320fd9369045cbe6a698813e/Ridges_Indychko.pdf).

### Пример работы

<figure>
   <figcaption>Исходное изображение (сосуды глазного дна)</figcaption>
  <img src="https://github.com/luseno4ek/edges-ridges/blob/9771bb2e4cf49f57c497c71ed39e0337f93addaf/in2.bmp" alt="Исходное изображение (сосуды глазного дна)" width="250"/>
</figure>
<figure>
   <figcaption>Результат выделения хребтовых структур</figcaption>
 <img src="https://github.com/luseno4ek/edges-ridges/blob/2617c11ea356304fb26e85edf026c2a651085b5e/myves.bmp" alt="Результат выделения хребтовых структур" width="250"/>
</figure>

<figure>
   <figcaption>Исходное изображение (ящерица)</figcaption>
  <img src="https://github.com/luseno4ek/edges-ridges/blob/044f0a474a9d7c830a4d8de42dc319956048b77e/Large_Scaled_Forest_Lizard.jpg" alt="Исходное изображение (ящерица)" width="250"/>
</figure>
<figure>
   <figcaption>Результат работы Canny Edge Detector</figcaption>
 <img src="https://github.com/luseno4ek/edges-ridges/blob/044f0a474a9d7c830a4d8de42dc319956048b77e/mylizzard.bmp" alt="Результат работы Canny Edge Detector" width="250"/>
</figure>

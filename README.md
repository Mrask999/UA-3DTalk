# UA-3DTalk
UNCERTAINTY-AWARE 3D EMOTIONAL TALKING FACE SYNTHESIS WITH EMOTION PRIOR DISTILLATION

<div style="text-align: justify;">
  Here include the core code of our project, code for three major modules in paper are included in corresponding branches.
  The pre-trained and backbone models being used (or referenced) are shown as below：
</div>


<div style="display: flex; justify-content: center;">
<table style="border-collapse: collapse; width: 90%; text-align: center;">
  <tr>
    <th style="border: 1px solid #ccc; padding: 8px;">Model name</th>
    <th style="border: 1px solid #ccc; padding: 8px;">Usage</th>
  </tr>

  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">
      <a href="https://github.com/Rudrabha/Wav2Lip" target="_blank">🌐 Wav2lip</a>
    </td>
    <td style="border: 1px solid #ccc; padding: 8px;">Prior Extraction(Lip Expert)</td>
  </tr>
  
  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">
      <a href="https://github.com/NeRF-3DTalker/NeRF-3DTalker-code" target="_blank">🌐 NeRF-3DTalker</a>
    </td>
    <td style="border: 1px solid #ccc; padding: 8px;">Prior Extraction(Style Net)</td>
  </tr>

  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">
      <a href="https://github.com/Fictionarry/TalkingGaussian" target="_blank">🌐 Talking-Gaussian</a>
    </td>
    <td style="border: 1px solid #ccc; padding: 8px;">Baseline</td>
  </tr>

  <tr>
    <td style="border: 1px solid #ccc; padding: 8px;">
      <a href="https://github.com/Vincent-ZHQ/CA-MSER" target="_blank">🌐 CA-MSER</a>
    </td>
    <td style="border: 1px solid #ccc; padding: 8px;">Emotion Distillation</td>
  </tr>
</table>
</div>

<div style="text-align: justify;">
  We appreciate their prior contributions and open-soure code!
</div>

## Environment and Pre-process
<div style="text-align: justify;">
  We implement our work with Python 3.7 on H20.
  The data pre-process follows works: Talking-Gaussian and NeRF-3DTalker. 
</div>


## **Training**
Before training, you should possess the check point of NeRF-3DTalker(including Wav2lip in it) and CA-MSER of your target video, links are shown as above. 
With them, you can have f_exp, f_tone in Prior Extration and emotional embedded audio feature (input of code-book in Emotion Distillation).
Put all of them in 
```bash
  ./data/<Your Actore Name>/ 
```
and run 
```bash
  ./scripts/train.sh
```


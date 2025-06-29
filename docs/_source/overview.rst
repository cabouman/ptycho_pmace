========
Overview
========

PMACE: Projected Multi-Agent Consensus Equilibrium
--------------------------------------------------

A reconstruction :math:`x` is subdivided into possibly overlapping components, :math:`v_j`, and distributed to multiple agents, :math:`F_j`, each of which acts to improve this local reconstruction. These local reconstructions are combined using a pixel-weighted averaging operator, :math:`G`, that reassembles the local components into a consistent global reconstruction.

.. figure:: PMACE_flow.png
   :width: 80%
   :align: center

   PMACE pipeline


Distributed Reconstruction and Application
------------------------------------------

**Ptychography** is an advanced computational imaging technique that can produce high-resolution images of nanoscale objects such as semiconductors and other manufactured nanomaterials. It works by moving a coherent X-ray probe across the object in overlapping patches and the far-field diffraction pattern associated with each scan position will be collected and used to recover the structure of the sample.

.. figure:: ptycho_imaging.png
   :width: 80%
   :align: center

   Ptychographic Imaging


Multi-mode Reconstruction in BM-PMACE
-------------------------------------

In ptychographic imaging, the illuminating probe can often consist of multiple spatial modes due to partial coherence or system imperfections. PMACE addresses this by enabling **multi-mode probe reconstruction**, where the algorithm simultaneously estimates several probe modes along with the object. Each mode is updated iteratively, and their contributions are weighted based on their relative energy. This approach allows for accurate modeling of the illumination, leading to improved reconstruction quality, especially in challenging experimental conditions.

.. figure:: PMACE_pipeline_probe.png
   :width: 80%
   :align: center

   PMACE probe reconstruction pipeline

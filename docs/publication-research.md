# Publishing the Mercury crater results: literature check, venues, and next steps

Research notes, written 2026-11-03. Every factual claim below links to the source I checked. Items
marked **unverified** were not confirmed, and need checking before anyone relies on them.

## Summary

| Claim | Verdict | Why |
|---|---|---|
| (a) Smooth 1–10 km N–S offset between the 2015 JMARS-era mosaic and the USGS basemap | **Weak: the offset is already published.** | Herrick et al. 2018 report a "varying, but typically ~5 km" offset between the 2013 MDIS v9 mosaic and the Becker et al. 2016 basemaps. The cause is documented: the earlier products used uncontrolled pointing. What's new here is only the measured smooth N–S field, which is useful but small. |
| (b) Blind adjudication corrects metrics against an incomplete catalog (raw P 0.35 → adjusted 0.70) | **Holds, with conditions.** | Others have checked new detections by hand (Silburt et al. 2019; Robbins, SwRI 2023), and the 2025 Mercury DL catalog reports recall only. A blind, source-hidden review that mixes catalog and detector circles is a real method contribution. But only 214 of 4,599 items are reviewed, and the only reviewer is the catalog's author. |
| (c) Strict "defined-rim" catalog agrees with the CNN at 11–23 km, while the CNN finds 4.4× more at 5–10 km | **Partly holds.** | A diverging count below ~10 km is *expected*. On Mercury, secondaries dominate below ~8–10 km, and Herrick 2018 excluded them. Analysts also differ by up to ±45% (Robbins et al. 2014). The new part is a quantified single-analyst-vs-CNN comparison, plus the strict criterion as an explanation. It needs Herrick 2018 as a third catalog to be convincing. |
| (d) New global ≥5 km Mercury catalog (54,598 detections) | **Not novel as a headline.** | Herrick et al. 2018 published a hand-made global >5 km catalog (~36K craters per their 2015 abstract). La Grassa et al. 2025 released a deep-learning global catalog of >6 million craters >400 m, public on figshare. |

**LPSC 2027:** abstracts are due **January 5, 2027, 11:59 p.m. US CST**, and the form opens in
November 2026 ([LPSC 2027](https://www.hou.usra.edu/meetings/lpsc2027/abstract-submission/)). An
abstract is realistic if it is framed around (b) and (c), and if it compares against Herrick 2018 and
La Grassa 2025.

---

## 1. Prior work

### 1a. Global / large-area Mercury crater catalogs

| Catalog | Diameter limit | Imagery | Size | Public? |
|---|---|---|---|---|
| Fassett et al. 2011, GRL, [doi:10.1029/2011GL047294](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2011GL047294) | ≥ 20 km, near-global | Mariner 10 + MESSENGER flybys | 8,775 craters, according to a secondary source ([Russian catalog paper](https://journals.rcsi.science/0320-930X/article/download/289440/387304)); **unverified** in the original | Herrick 2018 says it was "subsequently updated and maintained by lead author" ([Herrick 2018 §2](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2017JE005516)). I found no public download. |
| Herrick, Curran & Baer 2011, Icarus, [doi:10.1016/j.icarus.2011.06.021](https://sites.google.com/alaska.edu/robertherrick/resources/mercury-global-crater-database) | > 10 km (some smaller ones included) | Mariner 10 + MESSENGER flyby mosaic (USGS Map-a-Planet) | — | **Yes**: Excel/CSV on [Herrick's site](https://sites.google.com/alaska.edu/robertherrick/resources/mercury-global-crater-database) |
| **Herrick et al. 2018, JGR Planets, [doi:10.1029/2017JE005516](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2017JE005516)** | **> 5 km**, obvious secondaries excluded; morphology and 3-class degradation for > 10 km | MDIS v9 global mosaic (2013, USGS), then manually shifted to the Becker et al. 2016 basemaps | ~36K > 5 km and ~12K > 10 km ([2015 AGU abstract](https://ui.adsabs.harvard.edu/abs/2015AGUFM.P53A2097H)) | **Yes**: in the article's supporting information ([Acknowledgments](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2017JE005516)) |
| Kinczyk et al. 2020, Icarus 341, [doi:10.1016/j.icarus.2020.113637](https://profiles.wustl.edu/en/publications/a-morphological-evaluation-of-crater-degradation-on-mercury-revis/) | ≥ 40 km, degradation classes | MESSENGER | all craters ≥ 40 km | not checked |
| Morphological catalog of 12,365 craters ≥ 10 km (Russian, *Astronomicheskii Vestnik*) ([article](https://journals.rcsi.science/0320-930X/article/download/289440/387304)) | ≥ 10 km | MESSENGER | 12,365 | not checked |
| **La Grassa et al. 2025, Remote Sensing 17(19):3287, [mdpi.com/2072-4292/17/19/3287](https://www.mdpi.com/2072-4292/17/19/3287)**, catalog "ME6M300TGT" | > 400 m | Multimodal: MDIS visible mosaic + DTM + hillshade at 100 m/px ([EPSC-DPS 2025 abstract](https://meetingorganizer.copernicus.org/EPSC-DPS2025/EPSC-DPS2025-1625.html)) | "over 6 million craters" | **Yes**: [figshare doi:10.6084/m9.figshare.30187999.v1](https://doi.org/10.6084/m9.figshare.30187999.v1) |
| Szczech et al. 2024, impact structures and buried basins from stereo DTMs ([TU Berlin](https://depositonce.tu-berlin.de/items/4e2d612d-35ca-4711-8f98-f81c15eed33e/full)) | basins / large structures | MESSENGER stereo | — | not checked |
| Horstman 2016 (this project's thesis catalog) | ≥ 5 km, defined rims only | ~250 m/px MDIS mosaic in JMARS | 20,391 | — |

**Back-of-envelope comparison at ≥ 10 km.** The CNN finds 13,668 craters ≥ 10 km
(8,217 + 4,500 + 951; [global-run.md](global-run.md)). Herrick has ~12K > 10 km, and the thesis has
11,087. All three catalogs agree within ~20% at ≥ 10 km. They split apart only at 5–10 km: thesis
9.3K, Herrick ~24K (~36K − ~12K, from the 2015 abstract), CNN 40.9K. *This is my inference from the
cited counts, not a published comparison.*

### 1b. ML / automated crater detectors

| Work | Body / data | Reported result |
|---|---|---|
| Silburt et al. 2019, Icarus, [doi:10.1016/j.icarus.2018.06.022](https://arxiv.org/abs/1803.02192) ("DeepMoon") | Moon DEM; transferred to Mercury DEM | Recovers 92% of the human test-set craters and nearly doubles the detections. Manual check of 361 new craters: ~11% false positives. Mercury: "a large fraction" detected, with no number given. |
| La Grassa et al. 2025 (YOLOLens, multimodal) ([MDPI](https://www.mdpi.com/2072-4292/17/19/3287)) | Moon → Mercury by transfer learning | Recall vs Herrick at the south pole: 84% (352/418). Validation recall 58–95.7% across quadrangles ([EPSC-DPS 2025](https://meetingorganizer.copernicus.org/EPSC-DPS2025/EPSC-DPS2025-1625.html)). **I found no Mercury precision figure**: their validation is recall-based. |
| Martellato et al., EPSC 2024 ([abstract](https://meetingorganizer.copernicus.org/EPSC2024/EPSC2024-606.html)) | Mercury, DL catalog 1–167 km | SFD-based geologic analysis of the DL catalog |
| Jia et al. 2021, Remote Sensing 13(16):3193 ([record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsdoj,edsdoj.2469dc39d74e5b82350003053d577c)) | Moon; transfer to Mercury and Mars | Split-attention network; Mercury transfer (numbers not checked) |
| Arghavanian et al., EPSC 2026 ([abstract](https://meetingorganizer.copernicus.org/EPSC2026/EPSC2026-1132.html)) | Mercury, laser-altimeter topography | Preparatory work toward ML crater mapping for BepiColombo BELA |
| Robbins, SwRI IR&D 2023 ([SwRI](https://www.swri.org/what-we-do/internal-research-development/2023/earth-space/exploring-artificial-intelligence-machine-learning-advance-impact-crater-population-studies-19-r6364)) | YOLOv8 + human grading; tested on Mercury, Moon, Mars, Vesta, Ceres, Rhea | Workflow and grading interface; ">10×" speed-up is a projection, not a measurement |
| Lee & Hogan 2021, Computers & Geosciences, [doi:10.1016/j.cageo.2020.104645](https://arxiv.org/abs/2010.12520) | Mars | ~80% of known craters > 3 km found; 2–4% median location/diameter difference from other catalogs |
| Benedix et al. 2020, Earth and Space Science 7, e2019EA001005 ([Curtin](https://espace.curtin.edu.au/handle/20.500.11937/90815?show=full)) | Mars THEMIS, ±65°, YOLOv3 | 91% true-positive rate, ~15% diameter error. A possible calculation error and diameter bias are flagged by [Lee 2023](https://arxiv.org/pdf/2308.14650). |
| Tewari et al. 2023, review of DL crater detection ([arXiv:2310.07727](https://arxiv.org/html/2310.07727v1)) | all bodies | Covers more than 140 works up to Feb 2022 |
| Human reference catalogs: Robbins 2019 Moon ≥ 1–2 km, 1.3M craters, [doi:10.1029/2018JE005592](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins); Robbins & Hynek 2012 Mars ≥ 1 km, 384,343 craters, [doi:10.1029/2011JE003966](https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c4407-3f61bb0a70707f4d933364f567e59887b1524b5bd176077e7663fc53877d5fac3) | — | Ground truth used by most of the ML work above |

**Benchmarking honestly.** Our held-out T1–T3 figures:
- **Raw:** recall 0.82, precision 0.35, F1 0.49.
- **Verdict-adjusted:** P 0.70, R 0.88, F1 0.78; the 0.81 global-run figure uses a slightly different item set.

Silburt's 92% and Lee & Hogan's ~80% are recalls against a *human catalog*, and so is La Grassa's
84%. Our raw recall of 0.82 is the like-for-like number. None of these papers report a
blind-reviewed precision on Mercury, so the adjusted precision is the distinctive number. It needs
more review before it can carry weight (§7).

## 2. The registration offset

- **Already observed:** Herrick et al. 2018 compiled their catalog on the MDIS v9 mosaic (2013). They
  write that the Becker et al. 2016 basemaps "have a varying, but typically ~5 km, offset relative
  to the MDIS v9 mosaic". They moved craters by hand and still allow up to ~2 km of error
  ([Herrick 2018 §2](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2017JE005516)).
- **Documented cause: control.** The MDIS DDR notes say version 0 products used "best-determined"
  SPICE pointing and an ellipsoid shape model. Delivery 15 / v1 introduced c-smithed (controlled)
  kernels and the global DEM, which "reduced" the errors. The same notes say thermal effects can
  cause MDIS attitude errors of up to 350 µrad
  ([PDS MDIS DDR catalog](https://pds.nasa.gov/data/pds3/releases/backup/2016/201605.mes15hrdGrailNhsp/messMDS_3001_MDIS_DDR_DS.cat)).
- **The control network itself:** about 100,432 NAC/WAC-G images were bundle-adjusted with 12.6M
  control points ([Becker et al. 2016, LPSC abstract 2959](https://www.hou.usra.edu/meetings/lpsc2016/pdf/2959.pdf)).
  The USGS BDR 166 m v2 (2017) uses images controlled with the updated kernels
  ([search summary of the USGS BDR record](https://astrogeology.usgs.gov/search/map/mercury_messenger_mdis_global_basemap_bdr_166m)).
  I could not open that page directly (HTTP 403), so this is **partly unverified**.
- **Earlier mosaics:** flyby-era mosaics were tied to the Mariner 10 network and had ~2 km absolute
  errors ([phys.org 2009](https://phys.org/news/2009-12-messenger-team-global-mercury.html)). A
  later USGS flyby mosaic reports offsets "up to 10 km in some places"
  ([USGS NAC 1 km mosaic](https://astrogeology.usgs.gov/search/map/mercury_messenger_mdis_nac_global_mosaic_1km)).
- **Rotation model:** Stark et al. 2015 ([GRL, doi:10.1002/2015GL065152](https://elib.dlr.de/100866))
  revised the rotation parameters but largely confirmed the earlier pole. That is unlikely to be the
  main cause.

**Likely cause, then:** the 2015 JMARS mosaic was probably built before the Becker control network,
from uncontrolled pointing on an ellipsoid. The controlled basemap moved features by kilometres,
varying smoothly with the image footprints. **To pin it down:** identify the exact mosaic and
version JMARS served in 2015. It may well be MDIS v9 or v10, which would make our offset the same
one Herrick saw.

Two things are not in the literature:
- **The dominant N–S direction.** No source I found says the offset is mainly north–south.
- **A crater-derived, smooth correction field** for legacy catalogs.

The first is a hypothesis to test. One idea, not from a source: MESSENGER imaged the southern
hemisphere from higher altitude, which would turn the same pointing error into a larger ground
offset there.

## 3. How definitions and analysts change the SFD

- **Robbins et al. 2014,** Icarus, [doi:10.1016/j.icarus.2014.02.022](https://arxiv.org/abs/1404.1334):
  eight experts' counts differed by up to ~±45%, depending on diameter, terrain and bin counts. The
  highlands, which are old and degraded, showed the largest spread
  ([CosmoQuest summary](https://cosmoquest.org/x/?p=4287)). The paper recommends consensus counts by
  several people.
- **Secondaries on Mercury:**
  - Below ~8–10 km, secondaries outnumber primaries on Mercury
    ([Strom et al. 2015, RAA, doi:10.1088/1674-4527/15/3/009](https://arxiv.org/abs/1407.4521);
    [Chapman et al., EPSC-DPS 2011](https://meetingorganizer.copernicus.org/EPSC-DPS2011/EPSC-DPS2011-1497.pdf)).
  - Herrick 2018 finds the 5–10 km population shows "many background secondaries" around basins
    larger than 150 km.
  - Xiao's work on Hokusai secondaries also bears on this
    ([JGR 2016, doi:10.1002/2016JE005139](https://zendy.io/title/10.1002/2016je005139)).
- **Degradation schemes vary:** Herrick 2018 uses 3 classes; Kinczyk 2020 uses 5 (one 2022 abstract
  describes 6) ([NTRS](https://ntrs.nasa.gov/api/citations/20210026554/downloads/Mercury_Insolation_LPSC_2022.pdf)).
- **What it means for (c):** a strict "defined rim" catalog *should* fall below a detector at
  5–10 km, where degraded craters and secondaries dominate. Agreement at 11–23 km is then the
  expected like-for-like test. I found no published Mercury study that compares a single analyst's
  strict-criterion catalog with an ML catalog and attributes the gap to the criterion. That part is
  new, but modest.

## 4. LPSC 2027 (58th)

- **When and where:** March 15–19, 2027, The Woodlands, TX, and virtual
  ([LPSC 2027](https://www.hou.usra.edu/meetings/lpsc2027/)).
- **Dates:**
  - Abstract form opens in November 2026.
  - **Abstract deadline: January 5, 2027, 11:59 p.m. CST.**
  - Program out in early February 2027.

**Format rules.** These are from 2026; the 2027 rules aren't posted yet
([LPSC 2026 submission page](https://www.hou.usra.edu/meetings/lpsc2026/abstract-submission/)):
- 2 pages, PDF, US Letter, ≤ 3 MB, English, using the template or 1-inch margins.
- At most 2 abstracts per first author, of which only one can be oral.
- The presenter must be one of the first six authors.
- No revisions after the deadline.
- Abstracts are peer-reviewed by the program committee.

**Fees.** Also 2026 figures:
- **Submission:** $65 regular / $35 student, non-refundable.
- **Registration:** $650 in person / $475 virtual for professionals (early rate)
  ([LPSC 2026 registration](https://www.hou.usra.edu/meetings/lpsc2026/registration/)).

**Independent researchers:** the submission criteria set no affiliation requirement
([2026 criteria](https://www.hou.usra.edu/meetings/lpsc2026/abstract-submission/)). List
"Independent researcher" with a city as the affiliation.

**Topics.** These are from the 2026 list:
- **Primary:** "Mercury".
- **Secondary:** "Impacts".
- **"Planetary Spatial Data Infrastructure"** (posters only) suits a registration-offset poster.
- 2026 also had an AI special session ("The Agents of AI…"). The 2027 special sessions aren't posted
  yet; ideas were due October 18, 2026.

## 5. Venues for a full paper

| Venue | Fit | OA cost | Data rules | AI-assistance policy |
|---|---|---|---|---|
| **Planetary Science Journal** (AAS) | Best fit for Mercury science and a catalog | Gold OA, CC-BY. Priced per "quantum": $69.51 per 350 words, $61.40 per figure, table or data file ([AAS 2026 rates](https://journals.aas.org/article-charges-and-copyright)). E.g. 7,000 words + 10 figures + 4 tables ≈ **$2,250**. | **Unverified**; check the AAS data guide. Machine-readable tables count as quanta. | Disclose LLM use in manuscript preparation; AI can't be an author. Posted 9 Sep 2026 ([AAS guidelines](https://journals.aas.org/author-llm-guidelines)). |
| **Earth and Space Science** (AGU) | Good for methods or a dataset | Gold OA **$2,860**; Data Article and Method papers get **50% off** (≈ $1,430) ([AGU fees](https://www.agu.org/publications/authors/journals/publication-fees)) | Data Availability Statement in an "Open Research" section, with data and software cited ([AGU policies](https://www.agu.org/publications/authors/policies)) | Disclose AI used in writing, images, *or data collection/analysis*, in Methods: which tool and how ([AGU policies](https://www.agu.org/publications/authors/policies)) |
| **JGR: Planets** (AGU) | Strong for the science (Herrick 2018 is there) | Hybrid: $1,000 base fee without OA, or $4,190 OA ([AGU fees](https://www.agu.org/publications/authors/journals/publication-fees)) | as ESS | as ESS |
| **JGR: Machine Learning and Computation** (AGU) | Good for the adjudication method | $2,750 OA ([AGU fees](https://www.agu.org/publications/authors/journals/publication-fees)) | as ESS | as ESS |
| **Icarus** (Elsevier) | Classic venue for crater statistics (Silburt 2019, Kinczyk 2020) | Subscription route free; OA **$4,050** ([Icarus OA](https://www.sciencedirect.com/journal/icarus/publish/open-access-options)) | **unverified** | Separate AI declaration statement at submission ([Elsevier](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)) |
| **Computers & Geosciences** (Elsevier) | Methods and software; published Lee & Hogan 2021 | Subscription free; OA **$3,870** ([C&G OA](https://www.sciencedirect.com/journal/computers-and-geosciences/publish/open-access-options)) | **unverified** | as Icarus |
| Remote Sensing (MDPI) | Where La Grassa 2025 appeared | **unverified** | — | — |

**AI disclosure:** the code was written with Claude, and analysis help came from it too.
- **AGU** explicitly requires disclosure of AI used for data analysis.
- **AAS** covers manuscript preparation.
- **Elsevier** wants a declaration statement.

Plan a short Methods paragraph plus each journal's statement either way.

## 6. People who might be approached

This is a list only; nobody has been contacted. Recency is limited to what I verified.

| Researcher | Relevant recent work |
|---|---|
| Robert R. Herrick (U. Alaska Fairbanks) | Global > 5 km Mercury database ([JGR 2018](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2017JE005516)). The obvious comparison catalog; he also documented the mosaic offset. |
| Stuart J. Robbins (SwRI) | Inter-analyst variability ([Icarus 2014](https://arxiv.org/abs/1404.1334)), the lunar database ([2019](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins)), ML + human-grading crater work including Mercury ([SwRI 2023](https://www.swri.org/what-we-do/internal-research-development/2023/earth-space/exploring-artificial-intelligence-machine-learning-advance-impact-crater-population-studies-19-r6364)) |
| Riccardo La Grassa, Gabriele Cremonese, Elena Martellato, Cristina Re (INAF Padova) | ME6M300TGT DL catalog ([Remote Sensing 2025](https://www.mdpi.com/2072-4292/17/19/3287); [EPSC-DPS 2025](https://meetingorganizer.copernicus.org/EPSC-DPS2025/EPSC-DPS2025-1625.html)) |
| Caleb I. Fassett | ≥ 20 km catalog ([GRL 2011](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2011GL047294)), co-author of "Impact cratering of Mercury" ([2019 chapter](https://pubs.usgs.gov/publication/70211870)), associate editor on Herrick 2018 |
| Michelle Kinczyk, Louise Prockter | Global degradation states ≥ 40 km ([Icarus 2020](https://profiles.wustl.edu/en/publications/a-morphological-evaluation-of-crater-degradation-on-mercury-revis/)) |
| Lillian R. Ostrach | Co-author of Strom et al. 2015 ([arXiv](https://arxiv.org/abs/1407.4521)) and the 2019 Mercury cratering chapter ([USGS](https://pubs.usgs.gov/publication/70211870)) |
| Azar Arghavanian, Alexander Stark (MPS / DLR / TU Berlin) | Automated Mercury crater identification for BepiColombo ([EPSC 2026](https://meetingorganizer.copernicus.org/EPSC2026/EPSC2026-1132.html)); Stark also did Mercury rotation ([GRL 2015](https://elib.dlr.de/100866)) |

## 7. Recommendation

**Frame it as a three-catalog comparison with blind adjudication, not as "a new catalog".** Draft
title idea: *"What a strict hand-mapped Mercury crater catalog misses: CNN detection, blind
adjudication, and comparison with Herrick et al. (2018)."*

| | LPSC 2027 abstract | Journal paper (PSJ, or ESS methods) |
|---|---|---|
| Lead claims | (b) and (c); cite (a) as a known effect, with our measured field | (b) and (c) in full; (a) as a section; (d) as a released data product |
| Drop or soften | "first/new global catalog" | same |
| **Minimum extra work** | 1. Match against the Herrick 2018 SI catalog and the ME6M300TGT ≥ 5 km subset, giving counts per bin plus match rates. 2. Review a few hundred more items, mainly `cnn_reg_f2` 5–10 km detections; only 214 of 4,599 are reviewed now. 3. Find the exact 2015 JMARS mosaic version and cite Herrick's ~5 km offset. 4. State that the reviewer is the catalog author. | Everything in the LPSC column, plus: 5. **A second, independent reviewer** on a random subset (~300 items), reporting agreement (e.g. Cohen's κ); expected given ±45% expert spread (Robbins 2014). 6. Review samples outside T1–T3 (polar, the unregistered stripes). 7. Fix or explain the 30–45 km fine/coarse merge bump. 8. Separate secondaries from primaries at 5–10 km, or test against Herrick's secondary-excluded counts. 9. Release the catalog, model and code with a DOI (e.g. Zenodo) and a data-availability statement. 10. AI-use disclosure. |
| Second reviewer? | Nice to have; list it as a limitation | **Needed**: reviewers will ask, and a single reviewer who also made the catalog is the obvious objection |
| Cost | $65 submission plus registration (2026: $475 virtual / $650 in person) | PSJ ≈ $2–3k; ESS methods ≈ $1.4k; Icarus or C&G subscription route $0 |

A co-author from §6 would help with the second review and with credibility, especially Herrick (whose
catalog is the comparison) or Robbins (inter-analyst work).

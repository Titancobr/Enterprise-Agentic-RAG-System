# IP-SAKTI Sahayak — Official Sources Manifest

## Purpose
This manifest enumerates every authoritative, publicly available source that the IP-SAKTI corpus should draw from. Each entry includes the source URL, access method, update cadence, and licensing notes. This ensures:
- **Version tracking** — every answer can be traced to a specific document version
- **Compliance** — only open, non-paywalled sources are used without explicit user consent
- **Maintainability** — ingestion pipelines know exactly what to re-crawl and when

---

## 1. Indian Statutes & Rules (India Code)

| Source | URL | Access | Update Cadence | Notes |
|--------|-----|--------|----------------|-------|
| Patents Act, 1970 | `https://www.indiacode.nic.in/handle/123456789/2094` | HTML scrape / API | On amendment | Include 2024 Rules |
| Patents (Amendment) Rules, 2024 | `https://www.indiacode.nic.in/` | HTML | On notification | Section 3(p), Form 18A |
| Trade Marks Act, 1999 | `https://www.indiacode.nic.in/handle/123456789/1791` | HTML | On amendment | |
| Designs Act, 2000 | `https://www.indiacode.nic.in/handle/123456789/1383` | HTML | On amendment | |
| Geographical Indications Act, 1999 | `https://www.indiacode.nic.in/handle/123456789/1789` | HTML | On amendment | |
| Copyright Act, 1957 | `https://www.indiacode.nic.in/handle/123456789/2078` | HTML | On amendment | |
| Biological Diversity (Amendment) Act, 2023 | `https://www.indiacode.nic.in/` | HTML | On amendment | + 2024 Rules |
| Drugs & Cosmetics Act, 1940 | `https://www.indiacode.nic.in/handle/123456789/2155` | HTML | On amendment | First Schedule key |
| Drugs & Cosmetics Rules, 1945 | `https://www.indiacode.nic.in/` | HTML | On amendment | Phytopharmaceutical rules |
| Drugs & Magic Remedies Act, 1954 | `https://www.indiacode.nic.in/handle/123456789/2156` | HTML | On amendment | Advertising restrictions |
| FSS Act, 2006 | `https://www.indiacode.nic.in/handle/123456789/2115` | HTML | On amendment | |
| FSSAI Ayurveda Aahar Regulations, 2022/2024 | `https://www.fssai.gov.in/` | PDF | On notification | |
| Plant Variety Protection Act, 2001 | `https://www.indiacode.nic.in/` | HTML | On amendment | |

---

## 2. IP India Public Databases

| Database | URL | Access | Notes |
|----------|-----|--------|-------|
| InPASS (Patents) | `https://ipindia.gov.in/patent-search.htm` | Web search / CSV | Patent full-text, status |
| Trade Marks Registry | `https://ipindia.gov.in/trademark-search.htm` | Web search | TM classes, status |
| Designs Registry | `https://ipindia.gov.in/design-search.htm` | Web search | |
| GI Registry | `https://ipindia.gov.in/gi-registry.htm` | Web search | Registered GIs |

---

## 3. Biodiversity & ABS

| Source | URL | Access | Notes |
|--------|-----|--------|-------|
| National Biodiversity Authority | `https://nbaindia.org/` | HTML/PDF | ABS guidelines, forms |
| State Biodiversity Boards | Various | HTML/PDF | State-level rules |
| ABS Clearing-House | `https://absch.cbd.int/` | Web | International ABS records |

---

## 4. International Treaties & Frameworks

| Treaty / Instrument | URL | Access | Key Articles |
|---------------------|-----|--------|--------------|
| TRIPS Agreement | `https://www.wto.org/english/docs_e/legal_e/27-trips.pdf` | PDF | Art. 27, 30, 31 |
| CBD / Nagoya Protocol | `https://www.cbd.int/abs/` | HTML/PDF | Art. 6, 7, 15 |
| WIPO GRATK Treaty (2024) | `https://www.wipo.int/treaties/en/ip/genetic_resources/` | PDF | Art. 3–7, disclosure req. |
| PCT (Patent Cooperation Treaty) | `https://www.wipo.int/pct/en/` | HTML/PDF | International filing |
| Madrid System | `https://www.wipo.int/madrid/en/` | HTML | TM international |
| Hague System | `https://www.wipo.int/hague/en/` | HTML | Design international |
| Budapest Treaty | `https://www.wipo.int/treaties/en/registration/budapest/` | HTML | Microorganism deposits |

---

## 5. Pharmacopoeial & Standards

| Source | URL | Access | Notes |
|--------|-----|--------|-------|
| Ayurvedic Pharmacopoeia of India (API) | `https://pcimh.nic.in/` | PDF | Monographs, standards |
| Ayurveda Formulary of India (AFI) | `https://pcimh.nic.in/` | PDF | Formulations |
| Indian Pharmacopoeia | `https://ipc.gov.in/` | PDF | General monographs |

---

## 6. Traditional Knowledge Sources (Authorized Only)

| Source | URL | Access | Notes |
|--------|-----|--------|-------|
| TKDL Public Guidance | `https://tkdl.res.in/` | HTML | Public summaries only — **NO direct DB access** |
| WIPO TK Portal | `https://www.wipo.int/tk/en/` | HTML | Guidelines, not databases |

> ⚠️ **Important**: TKDL full database access is restricted. The assistant must **not claim** to "search TKDL" or "check TKDL prior art." Instead: "TKDL-aware prior-art guidance using authorized public sources."

---

## 7. Case Law (Indian)

| Source | URL | Access | Notes |
|--------|-----|--------|-------|
| Indian Kanoon | `https://indiankanoon.org/` | Search API | Patents, TM, GI cases |
| Supreme Court / High Court judgments | Various | HTML | Key IP precedents |

---

## 8. Export Market Regulatory Summaries (Secondary)

| Market | Source | Notes |
|--------|--------|-------|
| US FDA Botanical Drugs | `https://www.fda.gov/` | Guidance docs |
| EU HMPC / THMPD | `https://www.ema.europa.eu/` | Herbal medicine rules |
| ASEAN Harmonization | `https://asean.org/` | Regional guidelines |

---

## Ingestion Pipeline Requirements

1. **Version stamping** — Every chunk stored in Qdrant must carry:
   ```json
   {
     "source_id": "patents_act_1970",
     "source_version": "2024-03-15",
     "section": "3(p)",
     "url": "https://www.indiacode.nic.in/..."
   }
   ```

2. **Update detection** — Weekly cron to check `Last-Modified` headers / RSS feeds on India Code and IP India.

3. **Source priority** — Primary statutes > Rules > Case law > Commentary > Export summaries.

4. **Paid/Subscription sources** — Only ingested when user provides explicit, logged consent via `/sources/consent` endpoint.

---

## Corpus Schema (Qdrant Payload)

```json
{
  "text": "...chunk content...",
  "source_id": "patents_act_1970",
  "source_type": "statute",
  "jurisdiction": "INDIA",
  "category": "patent",
  "section": "3(p)",
  "version_date": "2024-03-15",
  "url": "https://www.indiacode.nic.in/...",
  "language": "en"
}
```

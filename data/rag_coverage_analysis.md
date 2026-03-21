# RAG Coverage Analysis: Test Cases vs. Available RAG Data

> **Purpose:** For each test case in both normal and adversarial benchmark sets, document whether the RAG system has ingested relevant information that could help the model respond successfully.

---

## RAG Data Sources

### For Sentiment (BERT 110M) & Spam (DistilBERT 66M)
**Mechanism:** In-memory knowledge base of labeled examples. Similarity voting via cosine distance.

**Sentiment KB (15 examples):**
| # | Text | Label |
|---|------|-------|
| 1 | Revenue exceeded analyst expectations by 8%. | positive |
| 2 | Quarterly profit margins expanded to a five-year high. | positive |
| 3 | Loan growth accelerated across all segments. | positive |
| 4 | Customer acquisition reached record levels in Q4. | positive |
| 5 | Net income rose 22% driven by higher interest rates. | positive |
| 6 | The firm reported a net loss for the third consecutive quarter. | negative |
| 7 | Operating expenses surged 20% due to regulatory fines. | negative |
| 8 | Credit quality deteriorated amid rising delinquencies. | negative |
| 9 | The bank warned of material weakness in internal controls. | negative |
| 10 | Provisions for bad debts increased sharply on weaker outlook. | negative |
| 11 | Total deposits remained flat compared to the prior period. | neutral |
| 12 | The board approved a routine extension of the credit facility. | neutral |
| 13 | Staffing levels were unchanged from last quarter. | neutral |
| 14 | The company filed its annual 10-K report on time. | neutral |
| 15 | Total branch count held steady at 4,200 locations nationwide. | neutral |

**Spam KB (15 examples):**
| # | Text | Label |
|---|------|-------|
| 1 | Congratulations! You have won a $500 gift card. Click here to claim. | spam |
| 2 | Your account has been suspended. Verify your identity immediately. | spam |
| 3 | Make money fast! Work from home and earn $10,000 per week. | spam |
| 4 | URGENT: Your PayPal account needs verification within 24 hours. | spam |
| 5 | Free trial! Get premium access to our service. No credit card required. | spam |
| 6 | Dear winner, you have been selected for an exclusive cash prize. | spam |
| 7 | Buy discount medications online. No prescription needed. | spam |
| 8 | Hi team, here are the meeting notes from yesterday's standup. | ham |
| 9 | Your order has shipped and will arrive by Friday. | ham |
| 10 | Please review the attached quarterly financial report. | ham |
| 11 | Reminder: your dentist appointment is tomorrow at 3pm. | ham |
| 12 | Thanks for submitting your pull request. I'll review it today. | ham |
| 13 | Your flight confirmation for March 22. Boarding pass attached. | ham |
| 14 | Monthly newsletter: product updates and engineering highlights. | ham |
| 15 | Hi, can we reschedule our 1-on-1 to Thursday afternoon? | ham |

### For Numerical & Financial Ratios (Llama2 7B)
**Mechanism:** ChromaDB vector store. 12 financial documents about Meridian National Bancorp, chunked at 300 words / 50 overlap, embedded with `all-MiniLM-L6-v2`. Top-3 chunks retrieved per query.

**Documents:** bank_annual_report_2023.txt, capital_ratios_2023.txt, consumer_banking_review.txt, credit_portfolio_2023.txt, financial_definitions.txt, interest_rate_analysis.txt, investment_banking_review.txt, market_outlook_2024.txt, operating_efficiency_2023.txt, regulatory_update.txt, revenue_segments_2023.txt, risk_management_2023.txt

**Critical design note:** Every numerical and financial ratio test case provides its **own self-contained data table** with numbers that are **different from Meridian's data** in the RAG documents. The RAG retrieval therefore injects potentially conflicting numbers from Meridian into the prompt alongside the test case's own table.

---

## NORMAL TEST SET

---

### Sentiment Analysis (s01 -- s20)

| ID | Text | Label | Category | Nearest KB Match(es) | RAG Coverage | Verdict |
|----|------|-------|----------|----------------------|-------------|---------|
| s01 | "Net interest income grew 12% driven by higher rates and loan growth." | positive | straightforward | KB#5: "Net income rose 22% driven by higher interest rates." | Very strong semantic match -- same pattern of income growth driven by rates. | COVERED |
| s02 | "Management expects headwinds from deposit competition to persist throughout 2024." | negative | domain_jargon | KB#10: "Provisions for bad debts increased sharply on weaker outlook." (weak match) | No KB example uses "headwinds" or "deposit competition." These are financial jargon terms. Nearest match is thematically distant. | NOT COVERED |
| s03 | "The company maintained its quarterly dividend of $0.50 per share." | neutral | subtle_neutral | KB#12: "The board approved a routine extension of the credit facility." | Moderate -- both describe routine corporate actions, but dividend maintenance is a different concept. Similarity voting should lean neutral. | PARTIALLY COVERED |
| s04 | "Credit costs increased significantly due to commercial real estate exposure." | negative | domain_specific | KB#8: "Credit quality deteriorated amid rising delinquencies." KB#10: "Provisions for bad debts increased sharply on weaker outlook." | Good match -- credit deterioration theme is well represented in KB. | COVERED |
| s05 | "Strong demand in core markets contributed to double-digit revenue growth." | positive | straightforward | KB#1: "Revenue exceeded analyst expectations by 8%." KB#3: "Loan growth accelerated across all segments." | Strong match -- revenue growth pattern is clearly represented. | COVERED |
| s06 | "Total assets remained relatively unchanged from the prior quarter." | neutral | subtle_neutral | KB#11: "Total deposits remained flat compared to the prior period." | Very strong match -- same "remained flat/unchanged" pattern for a financial aggregate. | COVERED |
| s07 | "The restructuring program resulted in $450M of one-time charges." | negative | domain_specific | KB#7: "Operating expenses surged 20% due to regulatory fines." | Partial -- both involve unexpected costs, but "restructuring charges" is a specific domain concept. KB has no example with "restructuring" or "one-time charges." | PARTIALLY COVERED |
| s08 | "Operating efficiency improved with the cost-to-income ratio declining to 52%." | positive | tricky_positive | No good match. KB#2: "Quarterly profit margins expanded" is closest but lacks the inversion trick. | NOT COVERED -- The word "declining" normally signals negative. Understanding that a declining cost-to-income ratio is positive requires domain knowledge not in the KB. KB has no example demonstrating this inversion. | NOT COVERED |
| s09 | "Provisions for loan losses rose sharply amid deteriorating credit quality." | negative | domain_jargon | KB#10: "Provisions for bad debts increased sharply on weaker outlook." | Near-exact match -- "provisions rose sharply" maps directly to KB#10. | COVERED |
| s10 | "The board approved a routine extension of the existing credit facility." | neutral | subtle_neutral | KB#12: "The board approved a routine extension of the credit facility." | Exact match (almost verbatim). | COVERED |
| s11 | "Revenue exceeded analyst consensus estimates by approximately 8%." | positive | straightforward | KB#1: "Revenue exceeded analyst expectations by 8%." | Near-exact match. | COVERED |
| s12 | "Non-performing loans increased to 1.2% of total loans from 0.9% a year ago." | negative | domain_specific | KB#8: "Credit quality deteriorated amid rising delinquencies." | Moderate -- NPL increase maps to credit deterioration theme, but no KB example specifically discusses NPL ratios. | PARTIALLY COVERED |
| s13 | "The company filed its annual 10-K report with the SEC on schedule." | neutral | subtle_neutral | KB#14: "The company filed its annual 10-K report on time." | Near-exact match. | COVERED |
| s14 | "Margin compression accelerated due to competitive deposit pricing pressures." | negative | domain_jargon | No good match. KB#7: "Operating expenses surged 20%" is thematically distant. | NOT COVERED -- "Margin compression" and "pricing pressures" are domain-specific terms with no semantic analog in the KB. | NOT COVERED |
| s15 | "Customer acquisition reached record levels with 2.3 million new accounts." | positive | straightforward | KB#4: "Customer acquisition reached record levels in Q4." | Near-exact match. | COVERED |
| s16 | "The bank warned of material weakness in internal controls over financial reporting." | negative | domain_specific | KB#9: "The bank warned of material weakness in internal controls." | Near-exact match. | COVERED |
| s17 | "Staffing levels were unchanged from last quarter at approximately 198,000 employees." | neutral | subtle_neutral | KB#13: "Staffing levels were unchanged from last quarter." | Near-exact match. | COVERED |
| s18 | "Fee income declined 15% year-over-year due to reduced trading activity." | negative | straightforward | KB#6: "The firm reported a net loss for the third consecutive quarter." KB#7: "Operating expenses surged 20%." | Partial -- the KB has negative financial examples but none about declining fee income or trading activity specifically. The word "declined" should still push toward negative via similarity. | PARTIALLY COVERED |
| s19 | "Return on tangible common equity improved to 18.2% from 16.9% in the prior year." | positive | domain_specific | KB#1: "Revenue exceeded analyst expectations by 8%." KB#2: "Quarterly profit margins expanded." | Partial -- no KB example about ROE/ROTCE improvement. The word "improved" should push toward positive, but it's a domain-specific metric. | PARTIALLY COVERED |
| s20 | "The acquisition is expected to be modestly dilutive to earnings in the first year before becoming accretive." | neutral | tricky_mixed | No good match. | NOT COVERED -- This sentence has mixed signals (dilutive = negative, accretive = positive, overall neutral). No KB example demonstrates this type of temporal signal balancing. | NOT COVERED |

**Summary:** 10/20 COVERED, 5/20 PARTIALLY COVERED, 5/20 NOT COVERED

---

### Spam Detection (sp01 -- sp20)

| ID | Text (subject line excerpt) | Label | Category | Nearest KB Match(es) | RAG Coverage | Verdict |
|----|----------------------------|-------|----------|----------------------|-------------|---------|
| sp01 | "Congratulations! You've won a $1,000,000 lottery prize!" | spam | obvious_spam | KB#1: "Congratulations! You have won a $500 gift card." KB#6: "Dear winner...exclusive cash prize." | Very strong match -- lottery/prize scam pattern is well covered. | COVERED |
| sp02 | "Q4 Board Meeting Agenda... attached is the agenda..." | ham | business | KB#8: "Hi team, here are the meeting notes from yesterday's standup." | Good match -- professional meeting context. | COVERED |
| sp03 | "URGENT: Your account has been compromised... verify your identity..." | spam | phishing | KB#2: "Your account has been suspended. Verify your identity immediately." KB#4: "URGENT: Your PayPal account needs verification." | Very strong match -- phishing pattern covered by two KB entries. | COVERED |
| sp04 | "Invoice #INV-2024-0847... consulting services... payment due..." | ham | business | KB#10: "Please review the attached quarterly financial report." | Moderate -- business document pattern. No KB example specifically about invoices. | PARTIALLY COVERED |
| sp05 | "Make $5000 per day from home!... secret system..." | spam | obvious_spam | KB#3: "Make money fast! Work from home and earn $10,000 per week." | Near-exact pattern match. | COVERED |
| sp06 | "Your prescription is ready for pickup..." | ham | notification | KB#11: "Reminder: your dentist appointment is tomorrow at 3pm." | Good match -- health/appointment notification pattern. | COVERED |
| sp07 | "Verify your PayPal account now... unusual login activity..." | spam | phishing | KB#4: "URGENT: Your PayPal account needs verification within 24 hours." | Near-exact match -- same brand, same tactic. | COVERED |
| sp08 | "Team standup notes - March 14..." | ham | business | KB#8: "Hi team, here are the meeting notes from yesterday's standup." | Near-exact match. | COVERED |
| sp09 | "Lose 30 pounds in 30 days guaranteed!... Doctors hate this..." | spam | obvious_spam | KB#7: "Buy discount medications online. No prescription needed." KB#3: "Make money fast!" | Moderate -- health scam pattern. KB has medication spam but not weight loss specifically. General spam signals (guaranteed, no effort) should still trigger. | COVERED |
| sp10 | "Your Amazon order has shipped..." | ham | notification | KB#9: "Your order has shipped and will arrive by Friday." | Near-exact match. | COVERED |
| sp11 | "Exclusive cryptocurrency investment opportunity... double your money..." | spam | scam | KB#6: "Dear winner...exclusive cash prize." KB#3: "Make money fast!" | Partial -- investment fraud pattern. No KB example about crypto or investment scams specifically. | PARTIALLY COVERED |
| sp12 | "Re: Project timeline update... adjusted milestones in Jira..." | ham | business | KB#12: "Thanks for submitting your pull request. I'll review it today." | Good match -- tech/project work context. | COVERED |
| sp13 | "Your Apple ID was used to sign in from a new device..." | spam | phishing | KB#2: "Your account has been suspended. Verify your identity." KB#4: "URGENT: Your PayPal account needs verification." | Good match -- account security phishing pattern. | COVERED |
| sp14 | "Dinner reservation confirmation... Saturday March 22 at 7:30 PM..." | ham | notification | KB#11: "Reminder: your dentist appointment is tomorrow at 3pm." KB#13: "Your flight confirmation for March 22." | Good match -- appointment/reservation confirmation. | COVERED |
| sp15 | "YOU HAVE BEEN SELECTED as our grand prize winner..." | spam | obvious_spam | KB#6: "Dear winner, you have been selected for an exclusive cash prize." | Near-exact match. | COVERED |
| sp16 | "Monthly engineering newsletter - March 2024..." | ham | newsletter | KB#14: "Monthly newsletter: product updates and engineering highlights." | Near-exact match. | COVERED |
| sp17 | "Important: Bank account verification required... 12 hours..." | spam | phishing | KB#2: "Your account has been suspended. Verify your identity immediately." | Strong match -- bank phishing pattern. | COVERED |
| sp18 | "PTO request approved... hand off critical tasks..." | ham | business | KB#15: "Hi, can we reschedule our 1-on-1 to Thursday afternoon?" | Moderate -- workplace admin context. No PTO-specific example but workplace communication pattern is covered. | PARTIALLY COVERED |
| sp19 | "Buy cheap medications online - 90% discount... No prescription..." | spam | obvious_spam | KB#7: "Buy discount medications online. No prescription needed." | Near-exact match. | COVERED |
| sp20 | "Can you review this PR?... authentication middleware changes..." | ham | business | KB#12: "Thanks for submitting your pull request. I'll review it today." | Very strong match -- PR/code review context. | COVERED |

**Summary:** 17/20 COVERED, 3/20 PARTIALLY COVERED, 0/20 NOT COVERED

---

### Numerical Reasoning (n01 -- n05 embedded in benchmark)

> **Note:** These test cases are embedded in `benchmark.py`. Each provides its own data table. RAG retrieves from the 12 Meridian documents. The numbers in the test tables differ from Meridian's data.

| ID | Question | Expected | Test Table Data | What RAG Likely Retrieves | Helpful? | Verdict |
|----|----------|----------|----------------|--------------------------|----------|---------|
| n01 | "What was the percentage change in total revenue from 2022 to 2023?" | 2.45% | Own table with segment revenues (Retail $14.2K, Wholesale $9.7K, Digital $3.1K) | Meridian's revenue: $48.7B (2023) vs $45.9B (2022), +6.2%. `bank_annual_report_2023.txt` or `revenue_segments_2023.txt` | **HARMFUL** -- Meridian's revenue change is 6.2%, test expects 2.45%. Retrieved numbers directly conflict with the test table. | NOT COVERED (harmful) |
| n02 | "Calculate the debt-to-equity ratio for 2023." | 1.10 | Own table: Total Debt $45.6B, Total Equity $41.5B | `financial_definitions.txt`: Meridian's D/E = 2.87x. Formula definition is helpful, but the specific number conflicts. | **MIXED** -- Formula definition is helpful, but Meridian's D/E of 2.87x could override the correct calculation from the test table (1.10). | PARTIALLY COVERED |
| n03 | "What is the efficiency ratio (OpEx/Revenue) for Q4 2023?" | 56.6% | Own table: Q4 OpEx $8.2B, Q4 Revenue $14.5B | `operating_efficiency_2023.txt`: Meridian efficiency ratio = 57.3%. `financial_definitions.txt`: efficiency ratio formula. | **MIXED** -- Formula is relevant but Meridian's 57.3% is dangerously close to the expected 56.6% and could be mistaken as the answer. | PARTIALLY COVERED |
| n04 | "Calculate the year-over-year change in net interest income." | 8.3% | Own table: NII 2023 $13.1B, NII 2022 $12.1B | `interest_rate_analysis.txt`: Meridian NII grew 11.3% ($26.6B to $29.6B). | **HARMFUL** -- Meridian's NII change is 11.3%, test expects 8.3%. Directly conflicting. | NOT COVERED (harmful) |
| n05 | "What was the total provision for credit losses as a percentage of total loans?" | 0.79% | Own table: Provision $5.8B, Total Loans $731.4B | `risk_management_2023.txt`: Provision $5.8B, `credit_portfolio_2023.txt`: Loans $731.4B. | **SPECIAL CASE** -- Test appears to use Meridian's actual numbers. RAG retrieval would provide the exact data needed. | COVERED |

**Summary:** 1/5 COVERED, 2/5 PARTIALLY COVERED (mixed helpful/harmful), 2/5 NOT COVERED (harmful)

---

### Financial Ratios (fr01 -- fr08)

| ID | Question | Expected | Test Table Numbers | What RAG Likely Retrieves | Helpful? | Verdict |
|----|----------|----------|--------------------|--------------------------|----------|---------|
| fr01 | "Decompose ROE using DuPont: (NI/Rev) x (Rev/Assets) x (Assets/Equity)" | 17.98% | NI $8.2B, Rev $56.3B, Assets $187.6B, Equity $45.6B | `financial_definitions.txt`: ROE formula. `bank_annual_report_2023.txt`: Meridian ROE=14.8%, NI=$13.4B, Rev=$48.7B. | **HARMFUL** -- Meridian's ROE is 14.8%; test expects 17.98%. Meridian's revenue ($48.7B) differs from test ($56.3B). Retrieved numbers conflict. Formula is in the question already. | NOT COVERED (harmful) |
| fr02 | "Calculate 3-year CAGR of revenue 2020-2023" | 10.17% | Rev 2023 $56.3B, Rev 2020 $42.1B | `bank_annual_report_2023.txt`: Meridian 2023 revenue $48.7B, 2022 $45.9B. No 2020 data in any document. | **HARMFUL** -- Meridian's revenue is $48.7B not $56.3B. No 2020 figures exist in RAG docs. Irrelevant retrieval. | NOT COVERED (harmful) |
| fr03 | "Working capital change Q3 to Q4" | 37.55% | Q4: CA $125.4B, CL $89.5B. Q3: CA $118.2B, CL $92.1B | No quarterly balance sheet data in RAG documents. No current assets/liabilities breakdown. | **IRRELEVANT** -- RAG docs have no quarterly balance sheet data and no current asset/liability breakdowns. | NOT COVERED |
| fr04 | "Debt-to-equity ratio (multi-component)" | 1.43 | ST Debt $12.4B, LT Debt $55.4B, Stock $18.2B, APIC $9.8B, RE $19.4B | `financial_definitions.txt`: D/E formula + Meridian D/E = 2.87x. | **HARMFUL** -- Formula is in the question already. Meridian's D/E of 2.87x directly conflicts with expected 1.43. | NOT COVERED (harmful) |
| fr05 | "Operating cash flow ratio = (NI + D&A - dAR + dAP) / CL" | 12.51% | NI $8.2B, D&A $3.4B, dAR $1.2B, dAP $0.8B, CL $89.5B | No cash flow statement data in RAG documents. No D&A, AR, or AP figures. | **IRRELEVANT** -- RAG docs have no cash flow components (D&A, receivables, payables). | NOT COVERED |
| fr06 | "Calculate gross, operating, net margins and spread" | 26.32% | Rev $56.3B, COGS $33.78B, OpEx $8.45B, Interest $3.8B, Tax $2.568B | No COGS or gross margin data for any entity in RAG docs. Banks don't typically report COGS. | **IRRELEVANT** -- RAG docs use bank accounting (NII + Non-II), not manufacturing accounting (COGS). No relevant data. | NOT COVERED |
| fr07 | "Sustainable growth rate = ROE x (1 - Payout Ratio)" | 12.59% | NI $8.2B, Equity $45.6B, Dividends $2.46B | `bank_annual_report_2023.txt`: Meridian NI=$13.4B, Dividends=$5.2B. `financial_definitions.txt`: ROE formula. | **HARMFUL** -- Meridian's NI ($13.4B) and dividends ($5.2B) differ from test data. Would compute wrong payout ratio. | NOT COVERED (harmful) |
| fr08 | "Interest coverage ratio (EBIT/Interest) improvement 2022-2023" | 21.74% | EBIT 2023 $15.2B, Int 2023 $3.8B, EBIT 2022 $13.8B, Int 2022 $4.2B | No EBIT figure in RAG docs. `interest_rate_analysis.txt` has interest expense context but for Meridian. | **IRRELEVANT** -- RAG docs don't report EBIT. Interest expense figures are for Meridian, not the test entity. | NOT COVERED |

**Summary:** 0/8 COVERED, 0/8 PARTIALLY COVERED, 8/8 NOT COVERED (5 harmful, 3 irrelevant)

---

### Striking Examples (fw01-fw04, rw01-rw02, hw01)

| ID | Text/Question | Type | RAG Source | Coverage | Verdict |
|----|---------------|------|------------|----------|---------|
| fw01 | "Management expects headwinds from deposit competition to persist throughout 2024." | Sentiment | Sentiment KB | No KB example with "headwinds." | NOT COVERED (by design -- fine-tuning wins) |
| fw02 | "Margin compression accelerated due to competitive deposit pricing pressures." | Sentiment | Sentiment KB | No KB example with "margin compression." | NOT COVERED (by design) |
| fw03 | "Operating efficiency improved with the cost-to-income ratio declining to 52%." | Sentiment | Sentiment KB | No KB example with domain-specific inversion. | NOT COVERED (by design) |
| fw04 | "The restructuring program resulted in $450M of one-time charges." | Sentiment | Sentiment KB | No KB example with "restructuring charges." | NOT COVERED (by design) |
| rw01 | "What is Meridian's CET1 capital ratio and how does it compare to regulatory minimums?" | Factual QA | `capital_ratios_2023.txt`: CET1 = 13.2%, regulatory min = 9.7% (4.5% + 3.2% SCB + 2.0% G-SIB) | **Exact data available.** This question is specifically about Meridian's data in the RAG docs. | COVERED (by design -- RAG wins) |
| rw02 | "What drove the decline in investment banking revenue according to the latest annual report?" | Factual QA | `investment_banking_review.txt`: IB fees down 3.0%, M&A advisory down 7.9% due to 12.4% decline in global M&A deal volume. | **Exact data available.** The answer is directly in the document. | COVERED (by design -- RAG wins) |
| hw01 | "Based on the annual report data, calculate the efficiency ratio change and explain what drove it." | Calculation + Factual | `operating_efficiency_2023.txt`: Efficiency ratio improved from 59.1% to 57.3%. Revenue growth outpaced expense growth. | **Exact data available.** Requires both retrieval (data) and reasoning (calculation). | COVERED (by design -- hybrid wins) |

---

## ADVERSARIAL TEST SET

---

### Adversarial Sentiment (as01 -- as30)

#### Knowledge Conflict (as01 -- as10)

| ID | Text (abbreviated) | Label | Adversarial Trick | Nearest KB Match(es) | Verdict |
|----|-------------------|-------|-------------------|----------------------|---------|
| as01 | "Record-breaking revenue of $12.8B... CEO warned... 15% workforce reduction." | negative | Positive numbers + negative outlook | KB#1 (revenue exceeded) pushes positive; KB#7 (expenses surged) pushes negative. | NOT COVERED -- KB has no mixed-signal examples. Conflicting matches will confuse voting. |
| as02 | "Shares surged 8%... analysts downgraded... unsustainable growth." | negative | Market reaction vs analyst consensus | KB#1 (revenue exceeded) pushes positive. No KB example of analyst downgrades. | NOT COVERED -- No concept of analyst action overriding market price. |
| as03 | "Leverage ratio improved to 1.2x... management 'deeply concerning.'" | negative | Improving metric + management concern | KB#2 (margins expanded) pushes positive. No KB example where management contradicts good numbers. | NOT COVERED -- No KB example of management sentiment overriding metrics. |
| as04 | "EPS beat consensus... all growth from one-time tax benefits." | negative | Strong earnings but unsustainable | KB#1/KB#5 (income rose, revenue exceeded) push positive. | NOT COVERED -- No concept of non-recurring vs recurring in KB. |
| as05 | "Losses narrowed significantly to $50M from $320M... expects breakeven." | positive | Negative word (losses) but improving trajectory | KB#6 (net loss) pushes negative due to "losses" keyword. | NOT COVERED -- No KB example of improving-trajectory-despite-loss pattern. |
| as06 | "Tier 1 capital robust at 14.2%... $2.1B goodwill impairment." | neutral | Strong capital + impairment charge | KB#9 (material weakness) pushes negative; no capital adequacy positive example. | NOT COVERED -- No example balancing strong fundamentals against one-time charges. |
| as07 | "Strategic pivot... 40% decline in legacy revenue... cloud ARR to offset." | neutral | Short-term negative during strategic transition | KB#6 (net loss) or KB#7 (expenses surged) push negative. | NOT COVERED -- No example of deliberate strategic transition being neutral. |
| as08 | "NIM expanded 35bps... 300% increase in CRE delinquencies." | negative | Improving margin vs credit deterioration | KB#2 (margins expanded) pushes positive; KB#8 (credit deteriorated) pushes negative. | NOT COVERED -- Mixed signals will split votes unpredictably. |
| as09 | "NPL ratio fell to 0.3%... provision coverage 450% vs industry 180%." | positive | Both metrics positive but large provision could signal concern | KB#10 (provisions increased sharply) pushes negative on "provision" keyword. | NOT COVERED -- No example where high provision is positive (strong coverage). |
| as10 | "Revenue flat at $8.2B... competitors reported double-digit declines." | positive | Flat = positive in relative context | KB#11 (deposits remained flat) pushes neutral. | NOT COVERED -- No example of relative outperformance concept. |

#### Noisy Retrieval (as11 -- as20)

| ID | Text (abbreviated) | Label | Adversarial Trick | Nearest KB Match(es) | Verdict |
|----|-------------------|-------|-------------------|----------------------|---------|
| as11 | "Net loss of $2.3B absorbing failed competitor... gained 4.2M depositors." | positive | Loss from value-accretive FDIC acquisition | KB#6 (net loss) pushes strongly negative. | NOT COVERED -- No concept of strategic loss being positive. |
| as12 | "Fee income fell 22% eliminating overdraft charges... customer retention up 15%." | positive | Revenue decline from positive strategic decision | KB#1 (revenue exceeded) pushes away; "fell 22%" pushes negative. | NOT COVERED -- No example of intentional revenue sacrifice being positive. |
| as13 | "CFO: 'not concerned about $4.5B unrealized loss in HTM portfolio.'" | neutral | Large unrealized loss but no intent to realize | KB#6 (net loss) or KB#9 (material weakness) push negative on "loss" keyword. | NOT COVERED -- No example of HTM accounting nuance. |
| as14 | "Provisions surged 180%... reflecting updated macro models not portfolio deterioration." | negative | Provisions surged but cause is model-driven | KB#10 (provisions increased sharply) -- aligns on negative. | PARTIALLY COVERED -- KB correctly identifies provision increase as negative, but misses the nuance that it's model-driven not actual deterioration. |
| as15 | "Mortgage originations collapsed 45%... in line with industry-wide decline." | negative | Negative but "in line with industry" context | No close KB match for mortgage volumes. | NOT COVERED -- No concept of industry-relative performance. |
| as16 | "Dividend increased 15th consecutive year, up 8%... cautious CRE outlook." | positive | Dividend growth positive; cautious outlook is noise | KB#1 (revenue exceeded) or KB#5 (net income rose) loosely match "increased." | PARTIALLY COVERED -- Positive signals in KB should lean correct direction, but cautious outlook may pull some negative matches. |
| as17 | "Non-interest expense up 12% from $2.8B tech investment... expected $1.5B annual savings by 2026." | neutral | Expense increase for investment with ROI | KB#7 (operating expenses surged 20%) pushes strongly negative. | NOT COVERED -- No concept of investment expense being neutral. |
| as18 | "Bonds upgraded to AA- by S&P... Moody's negative outlook." | neutral | Conflicting rating agency signals | No credit rating examples in KB. | NOT COVERED -- No rating agency examples at all. |
| as19 | "TBV/share declined 5% due to AOCI... excluding AOCI, TBV grew 8%." | neutral | Headline decline vs adjusted growth | No AOCI or book value examples in KB. | NOT COVERED -- No accounting adjustment examples. |
| as20 | "CEO resigned after internal investigation... COO appointed, widely respected." | negative | Negative event + positive successor | KB#9 (material weakness) loosely negative. No leadership change examples. | NOT COVERED -- No leadership transition examples. |

#### Out of Distribution (as21 -- as30)

| ID | Text (abbreviated) | Label | Adversarial Trick | Verdict |
|----|-------------------|-------|-------------------|---------|
| as21 | "First sustainability-linked bond, $2B at 15bps below conventional debt." | positive | ESG/green finance | NOT COVERED -- No ESG/green bond examples in KB. |
| as22 | "Quantum computing breakthrough... potentially obsoleting encryption infrastructure." | negative | Technology risk from adjacent field | NOT COVERED -- No technology risk examples. |
| as23 | "CBDC pilot processed 50K transactions... no revenue model established." | neutral | Novel fintech with unclear impact | NOT COVERED -- No CBDC/digital currency examples. |
| as24 | "Insurance subsidiary paid $890M catastrophe claims... 75% reinsurance recovery." | negative | Natural catastrophe insurance domain | NOT COVERED -- No insurance/reinsurance examples. |
| as25 | "AI fraud detection prevented $340M unauthorized transactions, 200% improvement." | positive | Technology operational metric | NOT COVERED -- No tech operational improvement examples. |
| as26 | "DeFi protocols offer 12% APY vs bank's 4.5%... deposit outflows accelerating." | negative | Fintech competitive threat | NOT COVERED -- No DeFi/competitive disruption examples. |
| as27 | "Carbon credit trading desk generated $45M... fastest-growing desk." | positive | Novel trading desk / ESG | NOT COVERED -- No carbon/ESG trading examples. |
| as28 | "Biometric auth reduced fraud 95%... implementation costs exceeded budget by 40%." | neutral | Great results but cost overrun | NOT COVERED -- No implementation cost-benefit examples. |
| as29 | "Crypto custody acquisition for $500M at 25x revenue... activist criticism." | negative | M&A in crypto / valuation controversy | NOT COVERED -- No crypto M&A or activist examples. |
| as30 | "Metaverse branch shuttered after 6 months... engagement fell short." | negative | Failed experimental initiative | NOT COVERED -- No failed initiative examples. |

**Summary:** 0/30 COVERED, 2/30 PARTIALLY COVERED, 28/30 NOT COVERED

---

### Adversarial Numerical (an01 -- an30)

> All adversarial numerical test cases provide self-contained data tables. RAG retrieves Meridian documents with different numbers.

#### Noisy Retrieval (an01 -- an15)

| ID | Question (abbreviated) | Expected | RAG Likely Retrieval | Verdict |
|----|----------------------|----------|---------------------|---------|
| an01 | "% change in total revenue 2022-2023" (3 segments with margin noise) | 7.57% | `revenue_segments_2023.txt`: Meridian revenue +6.2%. Different segments, different numbers. | NOT COVERED (harmful -- conflicting revenue figures) |
| an02 | "D/E ratio for 2023" (context floods with peer D/E values) | 1.17 | `financial_definitions.txt`: Meridian D/E = 2.87x. | NOT COVERED (harmful -- Meridian's 2.87x conflicts with 1.17) |
| an03 | "YoY revenue growth" (must sum quarterly then compare to 2022) | 6.87% | `bank_annual_report_2023.txt`: Meridian revenue 2023 $48.7B, +6.2%. | NOT COVERED (harmful -- Meridian's 6.2% is close but wrong answer) |
| an04 | "Diluted EPS" (must subtract preferred dividends) | 3.36 | `bank_annual_report_2023.txt`: Meridian EPS $9.82. | NOT COVERED (harmful -- Meridian's $9.82 is completely different) |
| an05 | "Overall revenue growth" (3 regions with FX/acquisition noise) | 6.97% | `bank_annual_report_2023.txt`: Meridian revenue +6.2%. | NOT COVERED (harmful -- 6.2% vs 6.97%) |
| an06 | "EBITDA and EBITDA margin" | 34.10% | No EBITDA in RAG docs. `operating_efficiency_2023.txt` has efficiency ratio. | NOT COVERED (irrelevant -- banks don't report EBITDA) |
| an07 | "Change in gross margin %" (restated vs original figures) | 1.99% | No gross margin data in RAG docs. | NOT COVERED (irrelevant) |
| an08 | "ROA %" | 4.37% | `bank_annual_report_2023.txt`: Meridian ROA = 1.12%. `financial_definitions.txt`: ROA formula + threshold. | NOT COVERED (harmful -- Meridian ROA 1.12% vs expected 4.37%) |
| an09 | "P/E and P/B ratios" (sector averages as distractors) | 13.61 | No P/E or P/B data in RAG docs. | NOT COVERED (irrelevant) |
| an10 | "EPS CAGR 2019-2023" (COVID outlier debate) | 8.29% | `bank_annual_report_2023.txt`: Meridian EPS $9.82 (2023), $8.67 (2022). No 2019 data. | NOT COVERED (irrelevant -- no historical EPS series) |
| an11 | "Coverage ratio (Allowance/NPL) change YoY" | 37.18% | `risk_management_2023.txt`: Meridian ACL/NPL = 2.42x. Different numbers. | NOT COVERED (harmful -- different coverage figures) |
| an12 | "Revenue per employee for most efficient segment" | 1329.41 | `operating_efficiency_2023.txt`: Meridian headcount 198,400. No segment headcount. | NOT COVERED (irrelevant -- no segment-level headcount) |
| an13 | "CET1, Tier 1, Total Capital ratios" | 13.00% | `capital_ratios_2023.txt`: Meridian CET1 = 13.2%, Tier 1 = 15.1%, Total = 17.7%. | NOT COVERED (harmful -- Meridian's ratios 13.2/15.1/17.7 vs test's different RWA/capital inputs) |
| an14 | "Annualized NIM for Q4 2023" | 3.20% | `interest_rate_analysis.txt`: Meridian Q4 NIM = 2.67%. | NOT COVERED (harmful -- Meridian's Q4 NIM 2.67% vs expected 3.20%) |
| an15 | "Total annual gross revenue from all products" | 63,660,000 | No product-level revenue data (checking/savings/cards by units). | NOT COVERED (irrelevant) |

#### Knowledge Conflict (an16 -- an20)

| ID | Question (abbreviated) | Expected | Verdict |
|----|----------------------|----------|---------|
| an16 | "Net income for 2023" (PPNR context distraction) | 7,332 | NOT COVERED (harmful -- Meridian NI = $13.4B, completely different) |
| an17 | "Adjusted net profit margin" (reported vs adjusted figures) | 14.81% | NOT COVERED (irrelevant -- no reported/adjusted distinction in RAG docs) |
| an18 | "Quick ratio" (analyst debate on excluding LT debt) | 0.92 | NOT COVERED (irrelevant -- no current asset/liability breakdown in RAG docs) |
| an19 | "5-year CAGR 2018-2023" (COVID outlier, context says use 2019) | 6.40% | NOT COVERED (irrelevant -- no 2018 or multi-year revenue series) |
| an20 | "Consolidated operating margin" (elimination column noise) | 23.91% | NOT COVERED (irrelevant -- no consolidation/elimination data) |

#### Out of Distribution (an21 -- an30)

| ID | Question (abbreviated) | Expected | Verdict |
|----|----------------------|----------|---------|
| an21 | "Total consolidated revenue in USD" (multi-currency FX conversion) | 39,598 | NOT COVERED (irrelevant -- no FX data in RAG docs) |
| an22 | "Probability-weighted expected portfolio loss" (stress test scenarios) | 2,710 | NOT COVERED (irrelevant -- no probabilistic scenario data) |
| an23 | "Approximate YTM for Series A bond" | 5.59% | NOT COVERED (irrelevant -- no bond-level data) |
| an24 | "Total expected loss across CDO tranches" | 294 | NOT COVERED (irrelevant -- no structured credit data) |
| an25 | "Net portfolio delta from 3 options" | 70.00 | NOT COVERED (irrelevant -- no options/derivatives data) |
| an26 | "Total expected credit loss across loan pools" (CECL) | 1,260.25 | NOT COVERED -- Meridian has aggregate loss data but not pool-level default/recovery rates in this format. |
| an27 | "Expected return of combined portfolio" (portfolio theory) | 9.98% | NOT COVERED (irrelevant -- no portfolio theory data) |
| an28 | "NPV at 9.5% discount rate" | 2,536.67 | NOT COVERED (irrelevant -- no project cash flow data) |
| an29 | "Highest ROE/PB ratio among 3 banks" | 13.44 | NOT COVERED (irrelevant -- no peer comparison data) |
| an30 | "Volume-weighted average price for the week" | 44.97 | NOT COVERED (irrelevant -- no trading/market data) |

**Summary:** 0/30 COVERED, 0/30 PARTIALLY COVERED, 30/30 NOT COVERED (15 harmful, 15 irrelevant)

---

### Adversarial Financial Ratios (afr01 -- afr25)

#### Noisy Retrieval (afr01 -- afr05)

| ID | Question (abbreviated) | Expected | Verdict |
|----|----------------------|----------|---------|
| afr01 | "Adjusted operating margin excl. restructuring" | 19.87% | NOT COVERED (irrelevant -- no COGS/SG&A/R&D breakdown in RAG docs; RAG docs use bank accounting) |
| afr02 | "ROTCE for 2023 and improvement vs 2022" | 3.00pp | NOT COVERED (harmful -- Meridian has ROE 14.8% but no ROTCE; goodwill amounts differ from test) |
| afr03 | "Efficiency ratio and PPNR" | 62.22% | NOT COVERED (harmful -- Meridian efficiency ratio is 57.3%, different from test's 62.22%) |
| afr04 | "Free cash flow yield" | 4.80% | NOT COVERED (irrelevant -- no FCF, market cap, or SBC data in RAG docs) |
| afr05 | "Loan-to-deposit ratio change" | 9.21pp | NOT COVERED (harmful -- Meridian has $731.4B loans, $934B deposits = 78.3% L/D; test uses different numbers) |

#### Knowledge Conflict (afr06 -- afr10)

| ID | Question (abbreviated) | Expected | Verdict |
|----|----------------------|----------|---------|
| afr06 | "3-component DuPont ROE decomposition" | 17.98% | NOT COVERED (harmful -- same numbers as fr01 but different total assets; Meridian's $1.24T assets vs test's $525B) |
| afr07 | "Reported net income growth rate" (context pushes adjusted) | 24.14% | NOT COVERED (harmful -- Meridian NI growth rate differs) |
| afr08 | "Tangible leverage ratio vs CET1 ratio difference" | 0.87pp | NOT COVERED (harmful -- Meridian has CET1 13.2%, SLR 6.4%; test uses different capital/asset figures) |
| afr09 | "Revenue per employee and OpEx per employee" | 284,343 | NOT COVERED (harmful -- Meridian has rev $48.7B / 198,400 FTE = $245,464; test uses $56.3B / 198,000 = $284,343) |
| afr10 | "Net income and net profit margin (NI / Total Revenue)" | 7.41% | NOT COVERED (harmful -- Meridian NII + Non-II = $48.7B, NI = $13.4B, margin ~27.5%; test data is completely different) |

#### Out of Distribution (afr11 -- afr20)

| ID | Question (abbreviated) | Expected | Verdict |
|----|----------------------|----------|---------|
| afr11 | "Segment with highest ROAC (Revenue/Capital)" | 117.78% | NOT COVERED (irrelevant -- no capital allocation by segment) |
| afr12 | "Enterprise Value and EV/EBITDA" | 9.98x | NOT COVERED (irrelevant -- no market cap, EBITDA, or minority interest) |
| afr13 | "4-year total shareholder return 2019-2023" | 57.07% | NOT COVERED (irrelevant -- no share price or dividend history) |
| afr14 | "Cumulative gap ratio at 12 months" (ALM) | -10.00% | NOT COVERED (irrelevant -- no maturity bucket asset/liability data) |
| afr15 | "Weighted average portfolio duration" | 4.57 | NOT COVERED (irrelevant -- no bond duration data) |
| afr16 | "Basic EPS with mid-year buyback/issuance adjustments" | 3.87 | NOT COVERED (harmful -- Meridian EPS $9.82 is different) |
| afr17 | "Fintech vs bank revenue per employee ratio" | 5.91x | NOT COVERED (irrelevant -- no fintech comparison data) |
| afr18 | "Total delinquency rate and net charge-off rate" | 2.90% | NOT COVERED (harmful -- Meridian has different delinquency/NCO breakdown) |
| afr19 | "Payout ratio = (Dividends + Buybacks) / FCF" | 90.00% | NOT COVERED (irrelevant -- no FCF in RAG docs) |
| afr20 | "Full-year NIM from total NII and avg earning assets" | 3.15% | NOT COVERED (harmful -- Meridian NIM is 2.68%; test has different NII/asset base) |

#### Knowledge Conflict continued (afr21 -- afr25)

| ID | Question (abbreviated) | Expected | Verdict |
|----|----------------------|----------|---------|
| afr21 | "Tangible Common Equity" | 30,700 | NOT COVERED (harmful -- Meridian CET1 is $98.4B; test equity/goodwill differ) |
| afr22 | "Basic and diluted EPS" (anti-dilutive securities trap) | 3.78 | NOT COVERED (harmful -- Meridian EPS $9.82) |
| afr23 | "ROA for Bank A vs Bank B" | 1.74% | NOT COVERED (harmful -- Meridian ROA 1.12%; test banks have different data) |
| afr24 | "Net interest spread (asset yield - liability cost)" | 2.41% | NOT COVERED (harmful -- Meridian's spread is 5.31% - 2.63% = 2.68%, different) |
| afr25 | "Degree of operating leverage" | 2.20 | NOT COVERED (irrelevant -- no variable/fixed cost breakdown) |

**Summary:** 0/25 COVERED, 0/25 PARTIALLY COVERED, 25/25 NOT COVERED (16 harmful, 9 irrelevant)

---

### Adversarial Spam (asp01 -- asp30)

#### Knowledge Conflict (asp01 -- asp10)

| ID | Text (abbreviated) | Label | Adversarial Trick | Nearest KB Match | Verdict |
|----|-------------------|-------|-------------------|------------------|---------|
| asp01 | "Action Required: Verify your identity... log in through official app or visit branch." | ham | Phishing language but legitimate | KB#2: "Account suspended. Verify identity." (spam) | NOT COVERED -- KB will vote spam due to "verify identity" match. No example of legitimate verification request. |
| asp02 | "Wire transfer of $48,500 initiated... call fraud hotline." | ham | Money transfer urgency like phishing | KB#4: "URGENT: PayPal needs verification." (spam) | NOT COVERED -- Urgency + money triggers spam KB matches. No bank alert examples. |
| asp03 | "Renew CFA certification at 40% off... cfainstitute.org/renew." | ham | Discount offer with deadline | KB#5: "Free trial! Premium access." (spam) | NOT COVERED -- "Discount", "limited time" trigger spam matches. No professional renewal examples. |
| asp04 | "Security alert: New device login... myaccount.google.com/security." | ham | Security alert = phishing pattern | KB#2: "Account suspended. Verify identity." (spam) | NOT COVERED -- Security alert language maps to spam KB. No legitimate security notification examples. |
| asp05 | "Password will expire in 3 days... press Ctrl+Alt+Delete." | ham | Password expiry = classic phishing | KB#4: "URGENT: PayPal needs verification." (spam) | NOT COVERED -- Password expiry urgency maps to phishing examples. No IT notification examples. |
| asp06 | "Netflix subscription payment failed... update payment method." | ham | "Update payment" = phishing signal | KB#2: "Account suspended. Verify identity." (spam) | NOT COVERED -- Payment failure + action required maps to phishing. No legitimate billing notice examples. |
| asp07 | "Tax refund notification - IRS... enter SSN and bank routing." | spam | IRS impersonation | KB#1: "Won $500 gift card." (spam) matches on prize/refund. KB#2: "Verify identity." (spam) | COVERED -- Prize/refund + identity verification signals correctly identify as spam. |
| asp08 | "Unusual activity on Schwab account... call 800-435-4000 or visit schwab.com." | ham | Account limitation but directs to call | KB#2: "Account suspended. Verify identity." (spam) | NOT COVERED -- Account security language maps to spam. No example distinguishing "call us" (ham) vs "click here" (spam). |
| asp09 | "Claim exclusive early access to GPT-5... first 1,000 respondents." | spam | AI hype exploitation | KB#6: "Dear winner, selected for exclusive cash prize." (spam) | COVERED -- "Selected", "exclusive", "limited" match spam KB patterns. |
| asp10 | "Board resolution - Emergency capital raise... confidential." | ham | Corporate urgency + confidentiality | KB#10: "Quarterly financial report." (ham) loosely matches. | PARTIALLY COVERED -- Financial context may lean toward ham KB matches, but urgency + confidentiality have no direct ham analogs. |

#### Noisy Retrieval (asp11 -- asp20)

| ID | Text (abbreviated) | Label | Adversarial Trick | Nearest KB Match | Verdict |
|----|-------------------|-------|-------------------|------------------|---------|
| asp11 | "Congratulations on your promotion!... team celebration Friday." | ham | "Congratulations" = spam trigger | KB#1: "Congratulations! Won $500." (spam) | NOT COVERED -- "Congratulations" is an exact semantic match to spam KB#1. Will vote spam for this ham email. |
| asp12 | "Selected for exclusive rewards program... call number on card." | ham | "Selected", "exclusive", "rewards" | KB#6: "Selected for exclusive cash prize." (spam) | NOT COVERED -- Multiple spam keyword matches. "Call number on your card" (legitimate) has no KB analog. |
| asp13 | "Free cloud storage upgrade - act now... upgrade is automatic." | ham | "Free", "upgrade", "act now" | KB#5: "Free trial! Premium access." (spam) | NOT COVERED -- "Free" + "upgrade" = strong spam match. "No action needed" caveat has no KB analog. |
| asp14 | "URGENT: Mandatory compliance training due Friday." | ham | "URGENT" + deadline + consequences | KB#4: "URGENT: PayPal needs verification." (spam) | NOT COVERED -- "URGENT" is a strong spam signal in KB. No compliance training examples. |
| asp15 | "Win a $500 Amazon gift card - Employee Appreciation Week." | ham | "Win", "gift card" = phishing | KB#1: "Won $500 gift card." (spam) | NOT COVERED -- Near-exact match to spam KB#1. Will vote spam. |
| asp16 | "Flight itinerary - Booking confirmation #AA4829." | ham | Travel + financial details + link | KB#13: "Flight confirmation for March 22." (ham) | COVERED -- Direct match to ham flight confirmation KB entry. |
| asp17 | "Re: Quick question about the merger docs... page 47 typo." | ham | Sensitive legal docs could be social engineering | KB#8: "Meeting notes from standup." (ham) or KB#10: "Financial report." (ham) | PARTIALLY COVERED -- Business communication pattern matches ham KB, but merger/legal specificity not covered. |
| asp18 | "System maintenance notification... trading platform Saturday 2-6 AM." | ham | System downtime for financial platform | KB#8: "Meeting notes from standup." (ham) | PARTIALLY COVERED -- Business context loosely matches, but no IT maintenance examples. |
| asp19 | "Important update to your benefits enrollment... closes March 31." | ham | Deadline + enrollment + "important update" | KB#4: "URGENT: PayPal needs verification." (spam) may trigger on urgency. | NOT COVERED -- "Important update" + deadline maps to phishing urgency. No HR benefits examples. |
| asp20 | "Invitation: Annual Shareholders Meeting... vote by proxy." | ham | External link + formal language | KB#10: "Financial report." (ham) loosely matches financial/corporate context. | PARTIALLY COVERED -- Corporate/financial context leans ham, but shareholder meeting has no direct KB analog. |

#### Out of Distribution (asp21 -- asp30)

| ID | Text (abbreviated) | Label | Adversarial Trick | Verdict |
|----|-------------------|-------|-------------------|---------|
| asp21 | "Recover your lost Bitcoin - guaranteed results... contingency basis." | spam | Targets crypto fraud victims | PARTIALLY COVERED -- "Guaranteed" maps loosely to spam patterns, but crypto recovery is OOD. |
| asp22 | "Research collaboration - MIT Sloan... $200K NSF grant." | spam | Academic spear-phishing | NOT COVERED -- Professional/academic tone maps to ham KB. Will likely misclassify as ham. |
| asp23 | "Invoice from your accountant... $12,000 additional deduction." | spam | Impersonates known contact + attachment | NOT COVERED -- "Invoice" and "attached" map to ham KB#10 (financial report). Will likely misclassify. |
| asp24 | "Regulatory filing assistance... disclosure deficiencies... SEC enforcement." | spam | Fear-based B2B spam with regulatory language | NOT COVERED -- Professional/financial language maps to ham KB. Fear element not covered. |
| asp25 | "LinkedIn: You appeared in 47 searches... upgrade to Premium." | spam | Mimics legitimate notification + upsell | NOT COVERED -- Notification pattern maps to ham. Upsell/promotional angle not in KB. |
| asp26 | "Vendor payment delay... resend payment using new wire instructions." | spam | BEC (Business Email Compromise) | NOT COVERED -- Business/invoice language maps to ham KB. BEC pattern not represented. |
| asp27 | "Your domain intelliswarm.ai expires in 48 hours." | spam | Domain renewal scam with real domain | NOT COVERED -- Urgency maps to spam, but domain renewal is OOD. Mixed signals. |
| asp28 | "Webinar: How AI is transforming financial auditing... ey-webinar-series.com." | spam | Impersonates Big 4 firm | NOT COVERED -- Professional/educational tone maps to ham KB. |
| asp29 | "Private equity co-investment - $5M minimum... Blackstone Partners." | spam | Investment scam using real PE firm name | NOT COVERED -- Financial/investment language maps to ham. No investment scam examples. |
| asp30 | "Your Zoom recording is ready... Q1 Financial Results." | spam | Fake Zoom notification | NOT COVERED -- Notification/meeting pattern maps to ham KB. Will likely misclassify. |

**Summary:** 3/30 COVERED, 5/30 PARTIALLY COVERED, 22/30 NOT COVERED

---

## GRAND SUMMARY

### Normal Test Set Coverage

| Test Section | Total | Covered | Partial | Not Covered | Coverage Rate |
|-------------|-------|---------|---------|-------------|---------------|
| Sentiment (s01-s20) | 20 | 10 | 5 | 5 | 50% full / 75% partial+ |
| Spam (sp01-sp20) | 20 | 17 | 3 | 0 | 85% full / 100% partial+ |
| Numerical (n01-n05) | 5 | 1 | 2 | 2 | 20% full / 60% partial+ |
| Financial Ratios (fr01-fr08) | 8 | 0 | 0 | 8 | 0% |
| Striking (rw/hw) | 3 | 3 | 0 | 0 | 100% |
| **TOTAL Normal** | **56** | **31** | **10** | **15** | **55% full / 73% partial+** |

### Adversarial Test Set Coverage

| Test Section | Total | Covered | Partial | Not Covered | Coverage Rate |
|-------------|-------|---------|---------|-------------|---------------|
| Adv. Sentiment (as01-as30) | 30 | 0 | 2 | 28 | 0% full / 7% partial+ |
| Adv. Numerical (an01-an30) | 30 | 0 | 0 | 30 | 0% |
| Adv. Financial Ratios (afr01-afr25) | 25 | 0 | 0 | 25 | 0% |
| Adv. Spam (asp01-asp30) | 30 | 3 | 5 | 22 | 10% full / 27% partial+ |
| **TOTAL Adversarial** | **115** | **3** | **7** | **105** | **3% full / 9% partial+** |

### Overall

| | Total | Covered | Partial | Not Covered | Harmful* |
|---|-------|---------|---------|-------------|----------|
| **ALL TEST CASES** | **171** | **34** | **17** | **120** | **~51** |

*\*Harmful = RAG retrieves Meridian data with different numbers than the test table, potentially misleading the model into using wrong figures.*

---

## KEY FINDINGS

### 1. Structural mismatch: Numerical/Financial Ratio tests vs RAG documents
All numerical (13 normal + adversarial) and financial ratio (33 normal + adversarial) test cases provide **self-contained data tables with their own numbers**. The RAG system retrieves Meridian documents with **completely different numbers**. In ~51 cases, this is actively **harmful** -- the model receives conflicting data and may use Meridian's figures instead of the test table's figures.

### 2. Sentiment KB too small for domain coverage
The 15-example sentiment KB covers basic patterns well (50% exact coverage) but completely misses: domain jargon ("headwinds", "margin compression"), context-dependent inversions ("declining cost-to-income = positive"), mixed signals, and novel financial domains (ESG, crypto, DeFi).

### 3. Spam KB vulnerable to adversarial patterns
The 15-example spam KB correctly classifies obvious spam and standard ham, but fails on: legitimate emails using phishing language (security alerts, password resets), sophisticated scams (BEC, impersonation), and the "congratulations" problem (workplace praise emails matching spam patterns).

### 4. Only "striking RAG wins" examples are properly designed for RAG
The 3 striking examples (rw01, rw02, hw01) are the **only** test cases where the question specifically asks about data that exists in the RAG documents. These demonstrate RAG's true value: answering questions about proprietary data not in the model's training set.

### 5. Adversarial set has near-zero RAG coverage by design
The adversarial test set (115 cases) achieves only 3% coverage, demonstrating that the current RAG knowledge base is insufficient for edge cases, mixed signals, novel domains, and sophisticated adversarial patterns.

# Academic Replication Study Agreement
## Lens v2.2 Code Search Benchmark

**Effective Date:** [Date]  
**Participating Institution:** [University/Lab Name]  
**Principal Investigator:** [Professor Name]  
**Sponsoring Organization:** [Your Organization]

---

## 1. Study Overview

### 1.1 Purpose
This agreement establishes terms for an independent replication study of the Lens v2.2 code search benchmark results published by [Your Organization]. The purpose is to validate the reproducibility and accuracy of our published performance claims through independent academic verification.

### 1.2 Scope of Work
**Participating Institution** agrees to:
- Execute the provided benchmark reproduction using supplied materials
- Follow prescribed methodology and measurement protocols
- Generate attested results in specified format
- Provide brief methodology confirmation report
- Allow public attribution of results (with proper academic credit)

### 1.3 Timeline
- **Study Duration:** Maximum 4 weeks from kit delivery
- **Technical Support Period:** 2 weeks of active engineering support
- **Payment Processing:** Within 10 business days of successful completion

---

## 2. Deliverables

### 2.1 Required Outputs
**Participating Institution** will provide:

1. **Primary Results File:** `hero_span_v22.csv` with reproduction results
2. **Environment Attestation:** SBOM and SHA256 checksums of execution environment  
3. **Methodology Confirmation:** 1-2 page report confirming adherence to protocols
4. **Technical Feedback:** Optional feedback on reproduction process and methodology

### 2.2 Success Criteria
Reproduction will be considered successful if results meet these acceptance gates:
- ✅ **CI Overlap:** Confidence intervals overlap with published results
- ✅ **Calibration Quality:** Max-slice ECE ≤ 0.02
- ✅ **Tail Behavior:** p99/p95 ratio ≤ 2.0  
- ✅ **Accuracy Tolerance:** nDCG@10 within ±0.1 percentage points

---

## 3. Provided Materials

### 3.1 Replication Kit Contents
**Sponsoring Organization** provides:
- Complete corpus manifest with SHA256 integrity checksums
- Docker-based reproduction environment with all dependencies
- SLA harness for precise timing measurement (150ms timeout)
- Automated validation scripts with acceptance criteria
- Comprehensive documentation and setup instructions
- Direct technical support from engineering team

### 3.2 Intellectual Property
- All provided materials remain property of **Sponsoring Organization**
- **Participating Institution** granted non-exclusive research use license
- Results may be used for academic publication with proper attribution
- No commercial use permitted without separate agreement

---

## 4. Compensation

### 4.1 Honorarium
**Sponsoring Organization** will pay **$2,500 USD** upon successful completion of study requirements, including:
- Delivery of all required outputs meeting success criteria
- Completion within agreed timeline
- Compliance with methodology protocols

### 4.2 Payment Terms
- Payment processed within 10 business days of acceptance
- Wire transfer or institutional check (Institution preference)
- All fees and taxes responsibility of **Participating Institution**
- No payment if success criteria not met (partial completion not compensated)

---

## 5. Academic Freedom & Publication

### 5.1 Research Independence
**Participating Institution** maintains complete academic freedom to:
- Analyze and critique provided methodology
- Identify limitations or concerns in experimental design
- Publish independent analysis of results
- Present findings at academic conferences

### 5.2 Attribution Rights
**Sponsoring Organization** may:
- Reference replication results in academic publications
- Include **Participating Institution** name in public leaderboard
- Acknowledge contribution in research papers and presentations
- Use results to support reproducibility claims

### 5.3 Publication Coordination
Both parties agree to:
- Coordinate timing of public announcements
- Share drafts of publications mentioning the replication
- Provide opportunity for comment before publication
- Ensure accurate representation of methodology and results

---

## 6. Confidentiality & Data Handling

### 6.1 Confidential Information
**Participating Institution** agrees to:
- Treat corpus data and queries as confidential research materials
- Use materials solely for agreed replication study purposes
- Not redistribute materials to third parties without permission
- Delete all materials within 30 days of study completion (unless separately agreed)

### 6.2 Public Disclosure
Following completion, both parties may publicly disclose:
- Final numerical results (nDCG@10 scores, confidence intervals)
- High-level methodology confirmation  
- Successful completion of replication study
- Academic attribution and collaboration details

---

## 7. Technical Support & Communication

### 7.1 Support Commitment
**Sponsoring Organization** provides:
- Direct email/Slack access to engineering team
- Video walkthrough of reproduction process
- Debugging assistance for technical issues
- Response time: 24 hours during business days

### 7.2 Communication Channels
- **Primary Contact:** [Engineering Lead Name] ([email])
- **Secondary Contact:** [Project Manager Name] ([email])
- **Emergency Contact:** [Phone number] (for critical technical issues)

---

## 8. Risk Management & Liability

### 8.1 Limitation of Liability
Neither party liable for indirect, incidental, or consequential damages. Maximum liability limited to honorarium amount ($2,500).

### 8.2 Force Majeure
Timeline extensions granted for circumstances beyond reasonable control (hardware failures, network outages, etc.).

### 8.3 Dispute Resolution
Good faith effort to resolve disputes informally. If unsuccessful, binding arbitration under [Jurisdiction] rules.

---

## 9. Termination

### 9.1 Termination Rights
Either party may terminate with 7 days written notice. **Participating Institution** retains rights to partial payment for completed work meeting success criteria.

### 9.2 Effect of Termination
Upon termination:
- **Participating Institution** ceases access to confidential materials
- **Sponsoring Organization** pays for any completed deliverables meeting criteria
- Both parties retain rights to publicly discuss completed work
- Publication rights remain in effect per Section 5

---

## 10. Signatures

**Participating Institution:**

______________________________________  
[Professor Name], Principal Investigator  
[Title]  
[University/Lab Name]  
Date: _____________

______________________________________  
[Administrator Name], Authorized Signatory  
[Title]  
[University/Lab Name]  
Date: _____________

**Sponsoring Organization:**

______________________________________  
[Your Name]  
[Your Title]  
[Your Organization]  
Date: _____________

---

**Appendix A: Technical Specifications**
- System requirements: Docker, 16GB+ RAM, 4+ cores, 50GB disk
- Expected execution time: 45-90 minutes  
- Corpus: 539 files, 2.3M lines, multiple programming languages
- Query volume: 48,768 queries across 5 language suites
- Evaluation method: SLA-bounded (150ms), pooled qrels, bootstrap sampling

**Appendix B: Success Metrics Detail**
- nDCG@10 tolerance: ±0.1 percentage points from published values
- Confidence interval overlap: CI bands must intersect with published CIs
- Quality gates: ECE ≤ 0.02, p99/p95 ≤ 2.0, error rate ≤ 0.1%
- Required attestation: SBOM with SHA256 checksums of execution environment

Generated: 2025-09-08T16:15:51.111Z  
Template Version: 1.0  
Ready for: Legal review and academic outreach

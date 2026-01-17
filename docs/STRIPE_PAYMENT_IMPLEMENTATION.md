# 💳 Stripe Payment Implementation Plan - DeepSeek Integration

## 🎯 Goal
Implement seamless payment flow where users pay **only after translation completes**, with charge amount = **actual API cost + $0.10 profit**.

---

## 📋 Prerequisites (What I Need From You)

### 1. Stripe Account Setup
- [ ] **Create Stripe Account**: https://dashboard.stripe.com/register (if you don't have one)
- [ ] **Get Test API Keys**:
  - Go to: https://dashboard.stripe.com/test/apikeys
  - Copy: `Publishable key` (starts with `pk_test_...`)
  - Copy: `Secret key` (starts with `sk_test_...`)
- [ ] **Enable Payment Intents API** (usually enabled by default)

### 2. DeepSeek API Cost Information
I need to know how to calculate costs:
- [ ] **Does DeepSeek API return usage/token data?** (Check their API response format)
- [ ] **Pricing Model**: 
  - Per token? (input + output tokens)
  - Per request?
  - Fixed per page?
- [ ] **Actual Pricing**:
  - Input tokens: $X per 1K tokens?
  - Output tokens: $X per 1K tokens?
  - Or fixed rate?

**If you don't know exact pricing**, I'll use a conservative estimate (~$0.001 per 1K tokens) and we can adjust later.

### 3. Profit Margin
- **Current Plan**: $0.10 fixed profit per transaction
- **Is $0.10 good?** 
  - ✅ For small docs (1-5 pages): Good (2-5% margin)
  - ⚠️ For large docs (50+ pages): Might be too low (0.1-0.2% margin)
  - **Recommendation**: Make it configurable or use percentage (e.g., 10% + $0.05 minimum)
- **Decision**: Keep $0.10 fixed OR use percentage? (I'll implement configurable)

---

## 🔄 Payment Flow (Detailed)

### Step-by-Step User Experience

```
1. User uploads document
   ↓
2. User selects "DeepSeek API" as translation provider
   ↓
3. User clicks "Translate" button
   ↓
4. ⚡ POPUP MODAL appears: "Payment Required"
   - Card input form (Stripe Elements)
   - Message: "Enter card details to proceed with DeepSeek translation"
   - Estimated cost display (optional)
   ↓
5. User enters card details and clicks "Pay & Translate"
   ↓
6. Backend: Create Payment Intent (authorize, don't capture yet)
   - Amount: Estimated ($5.00 max estimate for safety)
   - Status: "requires_capture"
   ↓
7. Frontend: Confirm payment (card authorized successfully)
   ↓
8. Backend: Start translation with DeepSeek
   - Track all API calls
   - Record token usage (input + output)
   ↓
9. Translation completes
   ↓
10. Backend: Calculate actual cost
    - Sum all DeepSeek API costs
    - Add $0.10 profit margin
    - Calculate final charge amount
    ↓
11. Backend: Capture payment for final amount
    - Update Payment Intent amount
    - Capture payment
    ↓
12. Frontend: Show success
    - "Translation complete! Charged: $X.XX"
    - Download button appears
```

---

## 🏗️ Architecture

### Backend Components

#### 1. Payment Service (`backend/app/services/payment_service.py`)
```python
class PaymentService:
    def create_payment_intent(amount, currency, metadata)
    def capture_payment_intent(payment_intent_id, amount)
    def calculate_deepseek_cost(api_responses)
    def calculate_final_charge(actual_cost, profit_margin=0.10)
    def refund_payment(payment_intent_id)
```

#### 2. Updated Translation Service
- Track DeepSeek API usage
- Return cost metadata with each translation

#### 3. Updated Job Processor
- After translation: Calculate cost → Capture payment
- Store payment info in job storage

#### 4. New API Endpoints
- `POST /api/v1/payment/create-intent` - Create payment intent
- `POST /api/v1/payment/capture/{job_id}` - Capture after translation
- `GET /api/v1/payment/status/{job_id}` - Check payment status

### Frontend Components

#### 1. Payment Modal (`frontend/src/components/PaymentModal.tsx`)
- Stripe Elements integration
- Card input form
- Loading states
- Error handling

#### 2. Updated Upload Flow (`frontend/src/app/page.tsx`)
- Show payment modal when DeepSeek selected
- Handle payment confirmation
- Show payment status

---

## 📝 Implementation Steps

### Phase 1: Backend Setup (Day 1)

#### Step 1.1: Install Stripe
```bash
cd backend
pip install stripe
```

Add to `requirements.txt`:
```
stripe>=7.0.0
```

#### Step 1.2: Environment Variables
Add to `backend/.env`:
```bash
# Stripe Configuration
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...  # Optional for webhooks

# Payment Settings
PAYMENT_PROFIT_MARGIN=0.10  # $0.10 fixed profit
PAYMENT_MINIMUM_CHARGE=0.50  # Minimum charge (covers Stripe fees)
PAYMENT_ESTIMATE_MULTIPLIER=5.00  # Max estimate for authorization
```

#### Step 1.3: Create Payment Service
**File**: `backend/app/services/payment_service.py`
- Stripe client initialization
- Payment Intent creation
- Payment capture
- Cost calculation
- Error handling

#### Step 1.4: Update Config
**File**: `backend/app/config.py`
- Add Stripe settings to `Settings` class

### Phase 2: Cost Tracking (Day 1-2)

#### Step 2.1: Update Translation Service
**File**: `backend/app/services/translation_service.py`
- Track DeepSeek API calls
- Record token usage (if available)
- Return cost metadata

#### Step 2.2: Update Job Processor
**File**: `backend/app/services/job_processor.py`
- Track cumulative costs during translation
- Store cost data in job storage
- After completion: Calculate total → Capture payment

### Phase 3: API Endpoints (Day 2)

#### Step 3.1: Payment Endpoints
**File**: `backend/app/main.py`
- `POST /api/v1/payment/create-intent`
- `POST /api/v1/payment/capture/{job_id}`
- `GET /api/v1/payment/status/{job_id}`

#### Step 3.2: Update Translate Endpoint
- Accept `payment_intent_id` parameter
- Link payment to job

### Phase 4: Frontend (Day 2-3)

#### Step 4.1: Install Stripe.js
```bash
cd frontend
npm install @stripe/stripe-js @stripe/react-stripe-js
```

#### Step 4.2: Create Payment Modal
**File**: `frontend/src/components/PaymentModal.tsx`
- Stripe Elements
- Card input
- Payment confirmation
- Loading/error states

#### Step 4.3: Update Main Page
**File**: `frontend/src/app/page.tsx`
- Show payment modal when DeepSeek selected
- Handle payment flow
- Show payment status

### Phase 5: Testing (Day 3)

- Test payment flow end-to-end
- Test with Stripe test cards
- Test error scenarios
- Test cost calculations

---

## 💰 Cost Calculation Logic

### DeepSeek API Cost
```python
def calculate_deepseek_cost(api_responses: List[dict]) -> float:
    """
    Calculate total DeepSeek API cost.
    
    Args:
        api_responses: List of API responses with usage data
        
    Returns:
        float: Total cost in USD
    """
    # DeepSeek pricing (UPDATE WITH ACTUAL RATES)
    INPUT_TOKEN_COST = 0.000001  # $0.001 per 1K tokens
    OUTPUT_TOKEN_COST = 0.000001  # $0.001 per 1K tokens
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for response in api_responses:
        usage = response.get("usage", {})
        total_input_tokens += usage.get("prompt_tokens", 0)
        total_output_tokens += usage.get("completion_tokens", 0)
    
    input_cost = (total_input_tokens / 1000) * INPUT_TOKEN_COST
    output_cost = (total_output_tokens / 1000) * OUTPUT_TOKEN_COST
    
    return input_cost + output_cost
```

### Final Charge Calculation
```python
def calculate_final_charge(actual_cost: float, profit_margin: float = 0.10) -> float:
    """
    Calculate final charge amount (cost + profit).
    
    Args:
        actual_cost: Actual API cost
        profit_margin: Profit margin in USD (default $0.10)
        
    Returns:
        float: Final charge amount
    """
    final = actual_cost + profit_margin
    
    # Apply minimum charge (covers Stripe fees ~$0.30 + $0.02 per transaction)
    MINIMUM_CHARGE = 0.50
    return max(final, MINIMUM_CHARGE)
```

---

## 🔒 Security Considerations

1. **Never expose secret keys** - Only publishable key in frontend
2. **Server-side validation** - All amounts calculated server-side
3. **Idempotency** - Use idempotency keys for payment operations
4. **Webhook verification** - Verify Stripe webhooks (optional but recommended)
5. **Error handling** - Handle payment failures gracefully
6. **Refund policy** - Define when to issue refunds

---

## 🧪 Testing Plan

### Test Scenarios

1. **Happy Path**
   - User enters card → Payment authorized → Translation completes → Payment captured

2. **Payment Failure**
   - Card declined → Show error → Allow retry

3. **Translation Failure**
   - Payment authorized → Translation fails → Issue refund

4. **Cost Variations**
   - Actual cost < estimate → Capture actual amount
   - Actual cost > estimate → Handle edge case (capture max, charge difference separately OR refund and create new payment)

5. **Edge Cases**
   - Network failures
   - Stripe API errors
   - Concurrent requests

### Stripe Test Cards
- **Success**: `4242 4242 4242 4242`
- **Decline**: `4000 0000 0000 0002`
- **3D Secure**: `4000 0025 0000 3155`

---

## 📦 Files to Create/Modify

### New Files
- `backend/app/services/payment_service.py`
- `frontend/src/components/PaymentModal.tsx`
- `frontend/src/hooks/usePayment.ts` (optional)

### Modified Files
- `backend/app/main.py` - Add payment endpoints
- `backend/app/models.py` - Add Payment model (optional, for database)
- `backend/app/config.py` - Add Stripe settings
- `backend/app/services/translation_service.py` - Track costs
- `backend/app/services/job_processor.py` - Capture payment after translation
- `backend/requirements.txt` - Add stripe
- `frontend/src/app/page.tsx` - Integrate payment modal
- `frontend/package.json` - Add Stripe dependencies
- `frontend/src/components/ProviderSelector.tsx` - Update text

---

## ⚠️ Important Notes

1. **Payment Authorization**: We authorize a maximum estimate ($5.00) but only capture the actual amount after translation
2. **Stripe Fees**: ~2.9% + $0.30 per transaction. Consider this in pricing
3. **Minimum Charge**: Recommend $0.50 minimum to cover fees
4. **Profit Margin**: $0.10 is good for small docs, consider percentage for large docs
5. **Refund Policy**: Define when refunds are issued (failed translations, user cancellations)

---

## 🚀 Next Steps

1. **You provide**:
   - Stripe test API keys
   - DeepSeek API pricing/usage format (or I'll use estimates)
   - Confirm profit margin preference ($0.10 fixed or percentage)

2. **I implement**:
   - Backend payment service
   - Cost tracking
   - Payment endpoints
   - Frontend payment modal
   - Integration with translation flow

3. **We test**:
   - End-to-end payment flow
   - Error scenarios
   - Cost calculations

---

## 💡 Profit Margin Analysis

### Is $0.10 Good?

**Current Plan**: $0.10 fixed per transaction

**Analysis**:
- ✅ **Small docs (1-5 pages)**: Good margin (2-5%)
- ⚠️ **Medium docs (10-20 pages)**: Acceptable (0.5-1%)
- ❌ **Large docs (50+ pages)**: Too low (0.1-0.2%)

**Recommendations**:

1. **Option A: Fixed $0.10** (Simpler)
   - Pros: Simple, predictable
   - Cons: Low margin on large docs
   - Best for: MVP, small-scale

2. **Option B: Percentage + Minimum** (Better)
   - Formula: `max(cost * 0.10, $0.10)` (10% or $0.10 minimum)
   - Pros: Scales better, fair for all sizes
   - Cons: Slightly more complex

3. **Option C: Tiered** (Best)
   - Small (<$1): $0.10
   - Medium ($1-5): $0.50
   - Large (>$5): $1.00
   - Pros: Best margins, covers fees
   - Cons: More complex

**My Recommendation**: Start with **Option B** (10% with $0.10 minimum) - it's simple and scales well.

---

**Ready to start implementation once you provide Stripe keys and pricing info!** 🚀


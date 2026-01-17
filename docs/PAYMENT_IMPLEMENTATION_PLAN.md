# 💳 Payment System Implementation Plan

## Overview
Implement Stripe payment integration for DeepSeek translations with **post-processing payment capture**:
1. User selects DeepSeek → Payment form appears
2. User enters credit card → Payment Intent created (authorized, not captured)
3. Document processed → Actual API costs calculated
4. Payment captured → Actual cost + $0.10 profit margin

---

## 📋 Prerequisites (What We Need From You)

### 1. Stripe Account Setup
- [ ] **Create Stripe Account**: https://dashboard.stripe.com/register
- [ ] **Get API Keys**:
  - Test Mode: `pk_test_...` (Publishable Key) and `sk_test_...` (Secret Key)
  - Live Mode: `pk_live_...` and `sk_live_...` (for production)
- [ ] **Enable Payment Intents API** (usually enabled by default)

### 2. DeepSeek API Cost Information
We need to know:
- [ ] **Pricing Model**: Per token? Per request? Fixed per page?
- [ ] **Input Token Cost**: e.g., $0.001 per 1K tokens
- [ ] **Output Token Cost**: e.g., $0.001 per 1K tokens
- [ ] **API Response Format**: Does DeepSeek return `usage` object with `prompt_tokens` and `completion_tokens`?

### 3. Environment Variables
Add to `backend/.env`:
```bash
# Stripe Configuration
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...  # For webhook verification (optional)

# Payment Settings
PROFIT_MARGIN=0.10  # $0.10 profit per transaction
MINIMUM_CHARGE=0.50  # Minimum charge amount (in USD)
```

---

## 🏗️ Architecture Overview

### Payment Flow
```
1. User selects DeepSeek
   ↓
2. Frontend: Show payment form (Stripe Elements)
   ↓
3. Backend: Create Payment Intent (authorize $X estimate)
   ↓
4. Frontend: Confirm payment (card authorized, not charged)
   ↓
5. Backend: Process document + track DeepSeek API costs
   ↓
6. Backend: Calculate final amount (actual_cost + $0.10)
   ↓
7. Backend: Capture payment for final amount
   ↓
8. Frontend: Show success + download link
```

### Components to Build

#### Backend
1. **Payment Service** (`app/services/payment_service.py`)
   - Create Payment Intent
   - Capture Payment Intent
   - Calculate costs
   - Handle refunds (if needed)

2. **Cost Tracking** (in `translation_service.py`)
   - Track DeepSeek API usage (tokens)
   - Calculate actual cost per request
   - Return cost metadata

3. **API Endpoints** (`app/main.py`)
   - `POST /api/v1/payment/create-intent` - Create payment intent
   - `POST /api/v1/payment/capture/{job_id}` - Capture payment after processing
   - `GET /api/v1/payment/status/{job_id}` - Check payment status

4. **Database Models** (`app/models.py`)
   - Add `Payment` model to track transactions
   - Link payments to jobs

#### Frontend
1. **Payment Form Component** (`components/PaymentForm.tsx`)
   - Stripe Elements integration
   - Card input, validation
   - Payment confirmation

2. **Payment Flow Integration** (`app/page.tsx`)
   - Show payment form when DeepSeek selected
   - Handle payment confirmation
   - Show payment status

---

## 📝 Implementation Steps

### Phase 1: Backend Setup

#### Step 1.1: Install Dependencies
```bash
cd backend
pip install stripe
```

Add to `requirements.txt`:
```
stripe>=7.0.0
```

#### Step 1.2: Create Payment Service
**File**: `backend/app/services/payment_service.py`
- `create_payment_intent(amount, currency, metadata)` - Create Stripe Payment Intent
- `capture_payment_intent(payment_intent_id, amount)` - Capture payment
- `calculate_cost(api_usage)` - Calculate DeepSeek API cost
- `refund_payment(payment_intent_id)` - Handle refunds

#### Step 1.3: Update Translation Service
**File**: `backend/app/services/translation_service.py`
- Modify DeepSeek translator to track:
  - Input tokens used
  - Output tokens used
  - API response metadata
- Return cost information with translation result

#### Step 1.4: Add Payment Endpoints
**File**: `backend/app/main.py`
- `POST /api/v1/payment/create-intent` - Create payment intent with estimated amount
- `POST /api/v1/payment/capture/{job_id}` - Capture payment after processing
- `GET /api/v1/payment/status/{job_id}` - Get payment status

#### Step 1.5: Update Job Processing
**File**: `backend/app/services/job_processor.py`
- After translation completes:
  - Calculate actual DeepSeek API cost
  - Add $0.10 profit margin
  - Capture payment for final amount
  - Update job status with payment info

#### Step 1.6: Database Models
**File**: `backend/app/models.py`
- Add `Payment` SQLAlchemy model:
  ```python
  class Payment(Base):
      id = Column(String, primary_key=True)
      job_id = Column(String, ForeignKey("jobs.id"))
      payment_intent_id = Column(String, unique=True)
      amount = Column(Float)
      currency = Column(String, default="usd")
      status = Column(String)  # pending, captured, failed, refunded
      actual_cost = Column(Float)
      profit_margin = Column(Float)
      created_at = Column(DateTime)
      captured_at = Column(DateTime, nullable=True)
  ```

### Phase 2: Frontend Setup

#### Step 2.1: Install Stripe.js
```bash
cd frontend
npm install @stripe/stripe-js @stripe/react-stripe-js
```

#### Step 2.2: Create Payment Form Component
**File**: `frontend/src/components/PaymentForm.tsx`
- Use Stripe Elements (`CardElement` or `PaymentElement`)
- Handle card input
- Submit payment intent confirmation
- Show loading states

#### Step 2.3: Update Main Page
**File**: `frontend/src/app/page.tsx`
- Show payment form when `translation_provider === "deepseek"`
- Handle payment confirmation
- Show payment status in job status
- Disable translation until payment confirmed

#### Step 2.4: Add Payment Status Display
- Show payment status: "Authorized", "Processing", "Charged", "Failed"
- Display final charge amount after processing

### Phase 3: Cost Calculation Logic

#### DeepSeek API Cost Calculation
```python
def calculate_deepseek_cost(api_response):
    """
    Calculate DeepSeek API cost from usage data.
    
    Args:
        api_response: DeepSeek API response with usage info
        
    Returns:
        float: Cost in USD
    """
    # DeepSeek pricing (update with actual rates)
    INPUT_TOKEN_COST = 0.000001  # $0.001 per 1K tokens
    OUTPUT_TOKEN_COST = 0.000001  # $0.001 per 1K tokens
    
    usage = api_response.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    
    input_cost = (input_tokens / 1000) * INPUT_TOKEN_COST
    output_cost = (output_tokens / 1000) * OUTPUT_TOKEN_COST
    
    total_cost = input_cost + output_cost
    return total_cost
```

#### Final Charge Calculation
```python
def calculate_final_charge(actual_api_cost, profit_margin=0.10):
    """
    Calculate final charge amount.
    
    Args:
        actual_api_cost: Actual DeepSeek API cost
        profit_margin: Profit margin in USD (default $0.10)
        
    Returns:
        float: Final charge amount
    """
    return actual_api_cost + profit_margin
```

---

## 🔄 Payment Flow Details

### Step-by-Step Flow

1. **User Selects DeepSeek**
   - Frontend: Show payment form
   - User enters card details

2. **Create Payment Intent** (Estimated Amount)
   ```javascript
   // Frontend calls backend
   POST /api/v1/payment/create-intent
   {
     "job_id": "uuid",
     "estimated_amount": 5.00,  // Estimate based on file size
     "currency": "usd"
   }
   
   // Backend returns
   {
     "client_secret": "pi_xxx_secret_xxx",
     "payment_intent_id": "pi_xxx"
   }
   ```

3. **Confirm Payment** (Authorize, Don't Capture)
   ```javascript
   // Frontend uses Stripe.js
   stripe.confirmCardPayment(client_secret, {
     payment_method: {
       card: cardElement,
     }
   })
   ```

4. **Start Translation**
   - Payment Intent status: `requires_capture`
   - Translation starts
   - DeepSeek API calls tracked

5. **Calculate Actual Cost**
   ```python
   # After translation completes
   actual_cost = calculate_deepseek_cost(api_responses)
   final_amount = actual_cost + 0.10  # Add profit margin
   ```

6. **Capture Payment**
   ```python
   # Backend captures payment
   POST /api/v1/payment/capture/{job_id}
   {
     "final_amount": 3.25  # Actual cost + $0.10
   }
   ```

7. **Handle Edge Cases**
   - If actual cost > estimated: Capture up to estimated, charge difference separately
   - If actual cost < estimated: Capture actual amount, refund difference
   - If payment fails: Retry or cancel job

---

## 🛡️ Security Considerations

1. **Never expose secret keys** - Only use publishable key in frontend
2. **Validate amounts server-side** - Never trust client-side calculations
3. **Webhook verification** - Verify Stripe webhooks (optional but recommended)
4. **Idempotency keys** - Use idempotency for payment operations
5. **Error handling** - Handle payment failures gracefully
6. **Refund policy** - Define refund policy for failed translations

---

## 📊 Database Schema

### Payment Table
```sql
CREATE TABLE payments (
    id VARCHAR PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    payment_intent_id VARCHAR UNIQUE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'usd',
    status VARCHAR(20) NOT NULL,
    actual_cost DECIMAL(10, 4) NOT NULL,
    profit_margin DECIMAL(10, 2) DEFAULT 0.10,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    captured_at TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);
```

### Job Table Updates
Add payment-related fields:
```sql
ALTER TABLE jobs ADD COLUMN payment_intent_id VARCHAR;
ALTER TABLE jobs ADD COLUMN payment_status VARCHAR;
ALTER TABLE jobs ADD COLUMN actual_cost DECIMAL(10, 4);
```

---

## 🧪 Testing Plan

### Test Scenarios

1. **Happy Path**
   - User selects DeepSeek → Enters card → Payment authorized → Translation completes → Payment captured

2. **Payment Failure**
   - Card declined → Show error → Allow retry

3. **Translation Failure**
   - Payment authorized → Translation fails → Refund payment

4. **Cost Variations**
   - Actual cost < estimate → Capture actual amount
   - Actual cost > estimate → Handle overage

5. **Edge Cases**
   - Network failures during payment
   - Stripe API errors
   - Concurrent payment attempts

---

## 📦 Files to Create/Modify

### New Files
- `backend/app/services/payment_service.py`
- `frontend/src/components/PaymentForm.tsx`
- `frontend/src/hooks/usePayment.ts` (optional)

### Modified Files
- `backend/app/main.py` - Add payment endpoints
- `backend/app/models.py` - Add Payment model
- `backend/app/services/translation_service.py` - Track costs
- `backend/app/services/job_processor.py` - Capture payment after processing
- `backend/app/config.py` - Add Stripe settings
- `backend/requirements.txt` - Add stripe
- `frontend/src/app/page.tsx` - Integrate payment flow
- `frontend/package.json` - Add Stripe dependencies

---

## ⚠️ Important Notes

1. **Stripe Test Mode**: Use test cards for development:
   - Success: `4242 4242 4242 4242`
   - Decline: `4000 0000 0000 0002`
   - 3D Secure: `4000 0025 0000 3155`

2. **DeepSeek API Response**: We need to verify the exact format of DeepSeek API responses to extract usage data correctly.

3. **Profit Margin**: $0.10 is fixed, but we can make it configurable via environment variable.

4. **Minimum Charge**: Consider adding a minimum charge (e.g., $0.50) to cover Stripe fees.

5. **Refund Policy**: Define when refunds are issued (failed translations, user cancellations, etc.).

---

## 🚀 Next Steps

1. **You provide**:
   - Stripe API keys (test mode)
   - DeepSeek API pricing details
   - Confirmation on profit margin ($0.10)

2. **I implement**:
   - Backend payment service
   - Cost tracking
   - Payment endpoints
   - Frontend payment form
   - Integration with translation flow

3. **We test**:
   - Test mode payments
   - Cost calculations
   - Payment capture
   - Error handling

---

## 📞 Questions for You

1. **DeepSeek Pricing**: What is the exact pricing model? (per token, per request, etc.)
2. **API Response**: Does DeepSeek return usage data in the response? If so, what's the format?
3. **Profit Margin**: Is $0.10 fixed, or should it be configurable?
4. **Minimum Charge**: Should we have a minimum charge amount?
5. **Refund Policy**: When should we issue refunds? (failed translations, user cancellations, etc.)
6. **Stripe Account**: Do you have a Stripe account, or should we set one up?

---

## 📚 Resources

- [Stripe Payment Intents](https://stripe.com/docs/payments/payment-intents)
- [Stripe Elements](https://stripe.com/docs/stripe-js/react)
- [Stripe Test Cards](https://stripe.com/docs/testing)
- [Stripe Webhooks](https://stripe.com/docs/webhooks)

---

**Ready to proceed once you provide the prerequisites!** 🚀


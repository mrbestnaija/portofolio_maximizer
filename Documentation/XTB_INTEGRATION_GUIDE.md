# 🏦 XTB API Integration Guide - Portfolio Maximizer v45
**Demo Account Trading Integration**

**Date**: October 19, 2025  
**Status**: Ready for Implementation  
**Broker**: XTB (X-Trade Brokers)  
**Account Type**: Demo Account

---

## 📊 EXECUTIVE SUMMARY

### XTB Integration Benefits
- ✅ **Demo Account**: Risk-free testing with real market conditions
- ✅ **Multiple Asset Classes**: Forex, Indices, Commodities, Stocks
- ✅ **High Leverage**: Up to 30:1 for Forex, 20:1 for Indices
- ✅ **Low Spreads**: Competitive spreads on major instruments
- ✅ **WebSocket API**: Real-time data and order execution
- ✅ **Regulated Broker**: EU-regulated with investor protection

### Supported Instruments
- **Forex**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Indices**: US500, US30, US100, UK100, GER30, FRA40
- **Commodities**: GOLD, SILVER, OIL, GAS
- **Stocks**: Major global stocks (optional)

---

## 🔧 IMPLEMENTATION PLAN

### **Phase 1: XTB Client Development (Week 3-4)**

#### **Day 15-17: Core XTB Client**
```python
# File: execution/xtb_client.py (NEW - 600 lines)
import os
import json
import websocket
import threading
from typing import Dict, List, Optional
from datetime import datetime
import logging

class XTBClient:
    """
    XTB API Client for Portfolio Maximizer v45
    Supports demo and live trading with comprehensive error handling
    """
    
    def __init__(self, mode='demo'):
        self.mode = mode  # 'demo' or 'live'
        self.server = "xtb-demo" if mode == 'demo' else "xtb"
        self.ws = None
        self.session_id = None
        self.connected = False
        
        # Load credentials from environment
        self.username = os.getenv('XTB_USERNAME')
        self.password = os.getenv('XTB_PASSWORD')
        
        if not self.username or not self.password:
            raise ValueError("XTB credentials not found in environment variables")
        
        # Initialize connection
        self._establish_connection()
    
    def _establish_connection(self):
        """Establish WebSocket connection to XTB API"""
        try:
            # Connect to XTB WebSocket
            self.ws = websocket.WebSocketApp(
                f"wss://ws.{self.server}.com/",
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            
            # Wait for connection
            self._wait_for_connection()
            
        except Exception as e:
            logging.error(f"Failed to connect to XTB: {e}")
            raise
    
    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        logging.info("Connected to XTB WebSocket")
        self.connected = True
        
        # Authenticate
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with XTB API"""
        auth_command = {
            "command": "login",
            "arguments": {
                "userId": self.username,
                "password": self.password,
                "appName": "PortfolioMaximizer",
                "appVersion": "1.0"
            }
        }
        
        self._send_command(auth_command)
    
    def _send_command(self, command: Dict):
        """Send command to XTB API"""
        if self.ws and self.connected:
            self.ws.send(json.dumps(command))
        else:
            raise ConnectionError("Not connected to XTB API")
    
    def place_order(self, signal: Dict) -> Dict:
        """
        Place order based on ML signal
        
        Args:
            signal: Dictionary containing:
                - symbol: Trading symbol (e.g., 'EURUSD')
                - action: 'BUY' or 'SELL'
                - volume: Position size
                - stop_loss: Stop loss level
                - take_profit: Take profit level
                - comment: Order comment
        
        Returns:
            Order result dictionary
        """
        try:
            # Validate signal
            self._validate_signal(signal)
            
            # Convert signal to XTB order format
            order = self._convert_signal_to_order(signal)
            
            # Send order command
            self._send_command(order)
            
            # Return order result
            return {
                "status": "success",
                "order_id": order.get("orderId"),
                "symbol": signal["symbol"],
                "action": signal["action"],
                "volume": signal["volume"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to place order: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_signal(self, signal: Dict):
        """Validate trading signal"""
        required_fields = ["symbol", "action", "volume"]
        for field in required_fields:
            if field not in signal:
                raise ValueError(f"Missing required field: {field}")
        
        if signal["action"] not in ["BUY", "SELL"]:
            raise ValueError("Action must be 'BUY' or 'SELL'")
        
        if signal["volume"] <= 0:
            raise ValueError("Volume must be positive")
    
    def _convert_signal_to_order(self, signal: Dict) -> Dict:
        """Convert ML signal to XTB order format"""
        return {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "cmd": 0 if signal["action"] == "BUY" else 1,  # 0=BUY, 1=SELL
                    "customComment": signal.get("comment", "ML Signal"),
                    "expiration": 0,  # GTC (Good Till Cancelled)
                    "order": 0,  # Market order
                    "price": 0,  # Market price
                    "sl": signal.get("stop_loss", 0),  # Stop loss
                    "symbol": signal["symbol"],
                    "tp": signal.get("take_profit", 0),  # Take profit
                    "type": 0,  # Open position
                    "volume": signal["volume"]
                }
            }
        }
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        command = {"command": "getMarginLevel"}
        self._send_command(command)
        # Handle response in _on_message
    
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        command = {"command": "getTrades", "arguments": {"openedOnly": True}}
        self._send_command(command)
        # Handle response in _on_message
    
    def close_position(self, position_id: int) -> Dict:
        """Close specific position"""
        command = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "cmd": 2,  # Close position
                    "order": position_id,
                    "type": 2  # Close position
                }
            }
        }
        self._send_command(command)
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            self._process_message(data)
        except Exception as e:
            logging.error(f"Error processing message: {e}")
    
    def _process_message(self, data: Dict):
        """Process incoming API messages"""
        if "status" in data:
            if data["status"]:
                logging.info(f"Command successful: {data.get('returnData', {})}")
            else:
                logging.error(f"Command failed: {data.get('errorCode', 'Unknown error')}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logging.error(f"XTB WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logging.info("XTB WebSocket connection closed")
        self.connected = False
    
    def _wait_for_connection(self, timeout=30):
        """Wait for WebSocket connection to be established"""
        import time
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.connected:
            raise ConnectionError("Failed to connect to XTB within timeout")
```

#### **Day 18-19: Order Management System**
```python
# File: execution/order_manager.py (ENHANCED - 450 lines)
class XTBOrderManager:
    """
    Enhanced order management system for XTB integration
    Handles order lifecycle, risk management, and position tracking
    """
    
    def __init__(self, xtb_client: XTBClient):
        self.xtb_client = xtb_client
        self.open_orders = {}
        self.position_history = []
        self.risk_manager = XTBRiskManager()
    
    def execute_signal(self, signal: Dict, portfolio: Dict) -> Dict:
        """Execute trading signal with comprehensive risk management"""
        
        # 1. Risk validation
        risk_check = self.risk_manager.validate_signal(signal, portfolio)
        if not risk_check["approved"]:
            return {
                "status": "rejected",
                "reason": risk_check["reason"],
                "timestamp": datetime.now().isoformat()
            }
        
        # 2. Position sizing
        adjusted_signal = self._calculate_position_size(signal, portfolio)
        
        # 3. Place order
        order_result = self.xtb_client.place_order(adjusted_signal)
        
        # 4. Track order
        if order_result["status"] == "success":
            self._track_order(order_result)
        
        return order_result
    
    def _calculate_position_size(self, signal: Dict, portfolio: Dict) -> Dict:
        """Calculate optimal position size based on Kelly criterion"""
        # Use existing portfolio_math.py for position sizing
        from etl.portfolio_math import calculate_kelly_fraction_correct
        
        # Get signal confidence and historical performance
        confidence = signal.get("confidence", 0.5)
        win_rate = signal.get("win_rate", 0.55)
        avg_win = signal.get("avg_win", 0.02)
        avg_loss = signal.get("avg_loss", 0.015)
        
        # Calculate Kelly fraction
        kelly_fraction = calculate_kelly_fraction_correct(win_rate, avg_win, avg_loss)
        
        # Apply confidence weighting
        position_size = kelly_fraction * confidence * portfolio["available_balance"]
        
        # Apply maximum position limits
        max_position = portfolio["available_balance"] * 0.02  # 2% max
        position_size = min(position_size, max_position)
        
        # Update signal with calculated position size
        signal["volume"] = position_size
        return signal
```

#### **Day 20-21: Risk Management Integration**
```python
# File: risk/xtb_risk_manager.py (NEW - 350 lines)
class XTBRiskManager:
    """
    XTB-specific risk management system
    Implements position limits, drawdown controls, and circuit breakers
    """
    
    def __init__(self):
        self.max_daily_loss = 0.05  # 5% maximum daily loss
        self.max_position_risk = 0.02  # 2% maximum risk per position
        self.max_open_positions = 10  # Maximum concurrent positions
        self.daily_pnl = 0.0
        self.open_positions = 0
    
    def validate_signal(self, signal: Dict, portfolio: Dict) -> Dict:
        """Validate trading signal against risk parameters"""
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss * portfolio["balance"]:
            return {
                "approved": False,
                "reason": "Daily loss limit exceeded"
            }
        
        # Check position count limit
        if self.open_positions >= self.max_open_positions:
            return {
                "approved": False,
                "reason": "Maximum open positions reached"
            }
        
        # Check position size limit
        position_risk = signal.get("volume", 0) / portfolio["balance"]
        if position_risk > self.max_position_risk:
            return {
                "approved": False,
                "reason": f"Position size too large: {position_risk:.2%}"
            }
        
        # Check instrument-specific limits
        instrument_check = self._check_instrument_limits(signal)
        if not instrument_check["approved"]:
            return instrument_check
        
        return {"approved": True, "reason": "Signal approved"}
    
    def _check_instrument_limits(self, signal: Dict) -> Dict:
        """Check instrument-specific risk limits"""
        symbol = signal["symbol"]
        
        # Forex limits
        if symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            max_leverage = 30
        # Index limits
        elif symbol in ["US500", "US30", "US100"]:
            max_leverage = 20
        # Commodity limits
        elif symbol in ["GOLD", "SILVER"]:
            max_leverage = 10
        else:
            max_leverage = 1
        
        # Check leverage
        leverage = signal.get("leverage", 1)
        if leverage > max_leverage:
            return {
                "approved": False,
                "reason": f"Leverage {leverage} exceeds limit {max_leverage} for {symbol}"
            }
        
        return {"approved": True, "reason": "Instrument limits OK"}
```

---

## 🔐 ENVIRONMENT SETUP

### **Step 1: Create .env File**
```bash
# Copy the template and fill in your XTB demo credentials
cp .env.template .env
```

### **Step 2: XTB Demo Account Setup**
1. **Register**: Visit [XTB Demo Account](https://xstation5.xtb.com/#/demo/loggedIn)
2. **Download**: Install XTB trading platform
3. **Credentials**: Note your demo username and password
4. **Update .env**: Fill in your credentials

### **Step 3: API Access**
```python
# Load credentials in your application
import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('XTB_USERNAME')
password = os.getenv('XTB_PASSWORD')
```

---

## 📊 TRADING INSTRUMENTS

### **Forex Pairs**
| Symbol | Description | Typical Spread | Leverage |
|--------|-------------|----------------|----------|
| EURUSD | Euro/US Dollar | 0.1 pips | 30:1 |
| GBPUSD | British Pound/US Dollar | 0.2 pips | 30:1 |
| USDJPY | US Dollar/Japanese Yen | 0.1 pips | 30:1 |
| USDCHF | US Dollar/Swiss Franc | 0.2 pips | 30:1 |
| AUDUSD | Australian Dollar/US Dollar | 0.2 pips | 30:1 |
| USDCAD | US Dollar/Canadian Dollar | 0.2 pips | 30:1 |
| NZDUSD | New Zealand Dollar/US Dollar | 0.3 pips | 30:1 |

### **Indices**
| Symbol | Description | Typical Spread | Leverage |
|--------|-------------|----------------|----------|
| US500 | S&P 500 | 0.5 points | 20:1 |
| US30 | Dow Jones | 1.0 points | 20:1 |
| US100 | NASDAQ | 1.0 points | 20:1 |
| UK100 | FTSE 100 | 0.8 points | 20:1 |
| GER30 | DAX | 0.8 points | 20:1 |
| FRA40 | CAC 40 | 0.8 points | 20:1 |

### **Commodities**
| Symbol | Description | Typical Spread | Leverage |
|--------|-------------|----------------|----------|
| GOLD | Gold Spot | 0.1 USD | 10:1 |
| SILVER | Silver Spot | 0.01 USD | 10:1 |
| OIL | Crude Oil | 0.05 USD | 10:1 |
| GAS | Natural Gas | 0.01 USD | 10:1 |

---

## 🎯 SUCCESS CRITERIA

### **Week 3-4 Targets**
- [ ] XTB WebSocket connection established
- [ ] Authentication working with demo account
- [ ] Order placement functional for all instrument types
- [ ] Position management operational
- [ ] Risk management integrated
- [ ] 50+ demo trades executed successfully

### **Performance Metrics**
- **Connection Stability**: >99% uptime
- **Order Execution**: <2 seconds average
- **Error Rate**: <1% failed orders
- **Risk Compliance**: 100% risk rule adherence

---

## 🛡️ RISK MANAGEMENT

### **Position Limits**
- **Maximum Position Size**: 2% of account balance
- **Maximum Daily Loss**: 5% of account balance
- **Maximum Open Positions**: 10 concurrent
- **Maximum Leverage**: Per instrument limits

### **Circuit Breakers**
- **Daily Loss Breaker**: Stop trading at 5% daily loss
- **Position Size Breaker**: Reject orders >2% of balance
- **Connection Breaker**: Pause trading on API disconnection
- **Error Breaker**: Stop trading after 5 consecutive errors

---

## 📚 REFERENCES

### **XTB Resources**
- [XTB Demo Platform](https://xstation5.xtb.com/#/demo/loggedIn)
- [XTB API Documentation](https://developers.x-station.xtb.com/)
- [XTB Trading Conditions](https://www.xtb.com/en/trading-conditions)

### **Implementation Files**
- `config/xtb_config.yml` - XTB configuration
- `execution/xtb_client.py` - XTB API client
- `execution/order_manager.py` - Order management
- `risk/xtb_risk_manager.py` - Risk management

---

**STATUS**: ✅ **READY FOR IMPLEMENTATION**  
**Next Action**: Create XTB demo account and begin Week 3-4 development  
**Timeline**: 2 weeks to full XTB integration  
**Risk Level**: Low (demo account only)

---

**Prepared by**: AI Development Assistant  
**Date**: October 19, 2025  
**Status**: XTB integration plan complete  
**Priority**: Week 3-4 implementation ready


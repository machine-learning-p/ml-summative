// main.dart - WAEMU Banking Risk Assessment Flutter App

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WAEMU Banking Risk Assessment',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: SplashScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    // Navigate to main screen after 3 seconds
    Future.delayed(Duration(seconds: 3), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomeScreen()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue[900],
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.account_balance,
              size: 80,
              color: Colors.white,
            ),
            SizedBox(height: 20),
            Text(
              'WAEMU Banking\nRisk Assessment',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 20),
            CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
            ),
          ],
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('WAEMU Banking Assessment'),
        backgroundColor: Colors.blue[800],
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Colors.blue[50]!, Colors.blue[100]!],
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Icon(
                        Icons.analytics,
                        size: 60,
                        color: Colors.blue[800],
                      ),
                      SizedBox(height: 16),
                      Text(
                        'Banking Risk Assessment',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.blue[800],
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        'Predict bank financial health using AI-powered analysis',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey[700],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => PredictionScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[800],
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: Text(
                  'Start Risk Assessment',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => InfoScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.grey[600],
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: Text(
                  'About the Model',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  String? _predictionResult;
  String? _errorMessage;

  // Controllers for all input fields
  final TextEditingController _countriesNumController = TextEditingController();
  final TextEditingController _yearController = TextEditingController();
  final TextEditingController _rirController = TextEditingController();
  final TextEditingController _sfsController = TextEditingController();
  final TextEditingController _infController = TextEditingController();
  final TextEditingController _eraController = TextEditingController();
  final TextEditingController _inlController = TextEditingController();
  final TextEditingController _debtController = TextEditingController();
  final TextEditingController _sizeController = TextEditingController();
  final TextEditingController _ccController = TextEditingController();
  final TextEditingController _geController = TextEditingController();
  final TextEditingController _psController = TextEditingController();
  final TextEditingController _rqController = TextEditingController();
  final TextEditingController _rlController = TextEditingController();
  final TextEditingController _vaController = TextEditingController();
  final TextEditingController _countriesController = TextEditingController();
  final TextEditingController _banksController = TextEditingController();

  // Replace with your actual API endpoint
  final String apiUrl = 'https://your-api-endpoint.com/predict';

  @override
  void initState() {
    super.initState();
    // Set default values
    _countriesController.text = 'Benin';
    _banksController.text = 'Default Bank';
    _yearController.text = '2023';
  }

  @override
  void dispose() {
    // Dispose all controllers
    _countriesNumController.dispose();
    _yearController.dispose();
    _rirController.dispose();
    _sfsController.dispose();
    _infController.dispose();
    _eraController.dispose();
    _inlController.dispose();
    _debtController.dispose();
    _sizeController.dispose();
    _ccController.dispose();
    _geController.dispose();
    _psController.dispose();
    _rqController.dispose();
    _rlController.dispose();
    _vaController.dispose();
    _countriesController.dispose();
    _banksController.dispose();
    super.dispose();
  }

  Future<void> _makePrediction() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _predictionResult = null;
      _errorMessage = null;
    });

    try {
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          'Content-Type': 'application/json',
        },
        body: json.encode({
          'countries_num': int.parse(_countriesNumController.text),
          'year': int.parse(_yearController.text),
          'rir': double.parse(_rirController.text),
          'sfs': double.parse(_sfsController.text),
          'inf': double.parse(_infController.text),
          'era': double.parse(_eraController.text),
          'inl': double.parse(_inlController.text),
          'debt': double.parse(_debtController.text),
          'size': double.parse(_sizeController.text),
          'cc': double.parse(_ccController.text),
          'ge': double.parse(_geController.text),
          'ps': double.parse(_psController.text),
          'rq': double.parse(_rqController.text),
          'rl': double.parse(_rlController.text),
          'va': double.parse(_vaController.text),
          'countries': _countriesController.text,
          'banks': _banksController.text,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _predictionResult =
          'Z-Score: ${data['predicted_zscore']}\n'
              'Risk Level: ${data['risk_level']}\n'
              'Assessment: ${data['interpretation']}\n'
              'Confidence: ${data['model_confidence']}';
        });
      } else {
        setState(() {
          _errorMessage = 'Error: ${response.statusCode} - ${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Network error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildInputField(
      String label,
      TextEditingController controller,
      String hint, {
        double? min,
        double? max,
        bool isInteger = false,
      }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: TextFormField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
          ),
          filled: true,
          fillColor: Colors.grey[50],
        ),
        keyboardType: isInteger
            ? TextInputType.number
            : TextInputType.numberWithOptions(decimal: true),
        validator: (value) {
          if (value == null || value.isEmpty) {
            return 'Please enter $label';
          }

          if (isInteger) {
            final intValue = int.tryParse(value);
            if (intValue == null) {
              return 'Please enter a valid integer';
            }
            if (min != null && intValue < min) {
              return 'Value must be at least ${min.toInt()}';
            }
            if (max != null && intValue > max) {
              return 'Value must be at most ${max.toInt()}';
            }
          } else {
            final doubleValue = double.tryParse(value);
            if (doubleValue == null) {
              return 'Please enter a valid number';
            }
            if (min != null && doubleValue < min) {
              return 'Value must be at least $min';
            }
            if (max != null && doubleValue > max) {
              return 'Value must be at most $max';
            }
          }
          return null;
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Banking Risk Prediction'),
        backgroundColor: Colors.blue[800],
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Colors.blue[50]!, Colors.white],
          ),
        ),
        child: Form(
          key: _formKey,
          child: ListView(
            padding: EdgeInsets.all(16),
            children: [
              Card(
                elevation: 2,
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Text(
                    'Enter Banking Metrics',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.blue[800],
                    ),
                  ),
                ),
              ),
              SizedBox(height: 16),

              // Basic Information
              _buildInputField('Country Number', _countriesNumController,
                  'Enter 1-8 for WAEMU countries', min: 1, max: 8, isInteger: true),
              _buildInputField('Year', _yearController,
                  'Enter year (2010-2025)', min: 2010, max: 2025, isInteger: true),

              // Risk Metrics
              _buildInputField('Risk Index Rating (RIR)', _rirController,
                  'Enter 0-10', min: 0, max: 10),
              _buildInputField('Solvency & Financial Stability (SFS)', _sfsController,
                  'Enter 0-100', min: 0, max: 100),
              _buildInputField('Inflation Rate (INF)', _infController,
                  'Enter -5 to 20', min: -5, max: 20),
              _buildInputField('Economic Risk Assessment (ERA)', _eraController,
                  'Enter 0-10', min: 0, max: 10),
              _buildInputField('Internationalization Level (INL)', _inlController,
                  'Enter 0-50', min: 0, max: 50),

              // Financial Metrics
              _buildInputField('Debt Level (DEBT)', _debtController,
                  'Enter 0-100', min: 0, max: 100),
              _buildInputField('Bank Size (SIZE)', _sizeController,
                  'Enter 0-30', min: 0, max: 30),
              _buildInputField('Capital Adequacy (CC)', _ccController,
                  'Enter 0-100', min: 0, max: 100),

              // Governance Metrics
              _buildInputField('Governance & Ethics (GE)', _geController,
                  'Enter 0-100', min: 0, max: 100),
              _buildInputField('Profitability & Sustainability (PS)', _psController,
                  'Enter 0-100', min: 0, max: 100),
              _buildInputField('Regulatory Compliance (RQ)', _rqController,
                  'Enter 0-100', min: 0, max: 100),
              _buildInputField('Liquidity Risk (RL)', _rlController,
                  'Enter 0-100', min: 0, max: 100),
              _buildInputField('Value Added (VA)', _vaController,
                  'Enter 0-100', min: 0, max: 100),

              // Categorical fields
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: TextFormField(
                  controller: _countriesController,
                  decoration: InputDecoration(
                    labelText: 'Country Name',
                    hintText: 'Enter country name',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    filled: true,
                    fillColor: Colors.grey[50],
                  ),
                ),
              ),

              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: TextFormField(
                  controller: _banksController,
                  decoration: InputDecoration(
                    labelText: 'Bank Name',
                    hintText: 'Enter bank name',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    filled: true,
                    fillColor: Colors.grey[50],
                  ),
                ),
              ),

              SizedBox(height: 24),

              // Predict Button
              ElevatedButton(
                onPressed: _isLoading ? null : _makePrediction,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[800],
                  foregroundColor: Colors.white,
                  padding: EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _isLoading
                    ? Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      ),
                    ),
                    SizedBox(width: 12),
                    Text('Predicting...'),
                  ],
                )
                    : Text(
                  'Predict',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),

              SizedBox(height: 24),

              // Results Display
              if (_predictionResult != null)
                Card(
                  elevation: 4,
                  color: Colors.green[50],
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.check_circle, color: Colors.green),
                            SizedBox(width: 8),
                            Text(
                              'Prediction Result',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.green[800],
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: 12),
                        Text(
                          _predictionResult!,
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.green[700],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

              if (_errorMessage != null)
                Card(
                  elevation: 4,
                  color: Colors.red[50],
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.error, color: Colors.red),
                            SizedBox(width: 8),
                            Text(
                              'Error',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.red[800],
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: 12),
                        Text(
                          _errorMessage!,
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.red[700],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class InfoScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('About the Model'),
        backgroundColor: Colors.blue[800],
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Colors.blue[50]!, Colors.white],
          ),
        ),
        child: ListView(
          padding: EdgeInsets.all(16),
          children: [
            Card(
              elevation: 4,
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'WAEMU Banking Risk Assessment',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue[800],
                      ),
                    ),
                    SizedBox(height: 16),
                    Text(
                      'This AI-powered application predicts bank financial health using the Z-Score metric for banks in the West African Economic and Monetary Union (WAEMU).',
                      style: TextStyle(fontSize: 16),
                    ),
                    SizedBox(height: 16),
                    Text(
                      'Z-Score Interpretation:',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text('• Z-Score ≥ 2.0: Low Risk (Excellent health)'),
                    Text('• Z-Score 1.5-2.0: Moderate Risk (Good health)'),
                    Text('• Z-Score 1.0-1.5: High Risk (Concerning)'),
                    Text('• Z-Score < 1.0: Very High Risk (Poor health)'),
                    SizedBox(height: 16),
                    Text(
                      'WAEMU Countries:',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text('1. Benin'),
                    Text('2. Burkina Faso'),
                    Text('3. Cote d\'Ivoire'),
                    Text('4. Guinea-Bissau'),
                    Text('5. Mali'),
                    Text('6. Niger'),
                    Text('7. Senegal'),
                    Text('8. Togo'),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
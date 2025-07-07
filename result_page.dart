import 'dart:convert';
import 'package:flutter/material.dart';

class ResultPage extends StatelessWidget {
  final String jsonString;

  ResultPage({required this.jsonString});

  @override
  Widget build(BuildContext context) {
    final data = json.decode(jsonString);
    final prediction = data['predicted_class'];
    final confidence = data['confidence'];

    return Scaffold(
      appBar: AppBar(title: Text('Prediction Result')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(30),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                "Species: $prediction",
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 15),
              Text(
                "Confidence: ${confidence.toStringAsFixed(2)}%",
                style: TextStyle(fontSize: 20),
              ),
              SizedBox(height: 30),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: Text("Try Another"),
              )
            ],
          ),
        ),
      ),
    );
  }
}

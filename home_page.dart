import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'result_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  final picker = ImagePicker();
  bool _loading = false;

  Future pickImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() => _image = File(pickedFile.path));
    }
  }

  Future<void> uploadImage() async {
    if (_image == null) return;

    setState(() => _loading = true);

    var request = http.MultipartRequest(
      'POST',
      Uri.parse(
          'https://paw-scan.onrender.com/predict'), // Use 10.0.2.2 for Android emulator
    );
    request.files.add(
      await http.MultipartFile.fromPath('image', _image!.path),
    );

    var response = await request.send();

    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultPage(jsonString: respStr),
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${response.statusCode}')),
      );
    }

    setState(() => _loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('PawScan')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            _image != null
                ? Image.file(_image!, height: 250)
                : Container(
                    height: 250,
                    color: Colors.grey[200],
                    child: const Center(child: Text("No Image Selected")),
                  ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed: () => pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Camera"),
                ),
                const SizedBox(width: 10),
                ElevatedButton.icon(
                  onPressed: () => pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo),
                  label: const Text("Gallery"),
                ),
              ],
            ),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: _loading ? null : uploadImage,
              style: ElevatedButton.styleFrom(
                padding:
                    const EdgeInsets.symmetric(horizontal: 50, vertical: 15),
              ),
              child: _loading
                  ? CircularProgressIndicator(color: Colors.white)
                  : Text("Predict"),
            ),
          ],
        ),
      ),
    );
  }
}

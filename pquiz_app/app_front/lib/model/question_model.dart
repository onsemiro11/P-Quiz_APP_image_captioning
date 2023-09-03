//Question 객체 정의
class Question{
  final String id;
  final String image_path;
  final Map<String, bool> options;

  Question({
    required this.id,
    required this.image_path,
    required this.options,
  });
  @override
  String toString() {
    return 'Question(id: $id, title: $image_path, options: $options)';
  }
}




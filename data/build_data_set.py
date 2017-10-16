#-*-coding:utf-8-*-
import os
import pdb
import random

# XX是不是你们投资的？
# XX公司是你们投资的吗？
# XX公司是你们什么时候投资的？
# 谁投资的XX公司？
# 你们有没有投资XX公司？
# 你们有投资XX公司吗？
# 你们投资了XX公司吗？
# XX公司被你们投资了吗？
_companys_zh = ['七牛', '名片提名王','薄荷', '英语流利说']
_companys_en = ['UCloud', 'Stringkly', 'Ruff', 'Glow']
def build_data_set_zh():
	dataset = []
	# XX是不是你们投资的？
	tag_tp = '\tO\tO\tO\tB-investor\tI-investor\tB-relation\tI-relation\tO\tO'
	for company in _companys_zh:
		_input = '\t'.join(list(company + '是不是你们投资的？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = '\t'.join(_tag_list) + tag_tp
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# XX是你们投资的吗？
	tag_tp = '\tO\tB-investor\tI-investor\tB-relation\tI-relation\tO\tO\tO'
	for company in _companys_zh:
		_input = '\t'.join(list(company + '是你们投资的吗？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = '\t'.join(_tag_list) + tag_tp
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# XX是你们什么时候投资的？
	tag_tp = '\tO\tB-investor\tI-investor\tB-ask-time\tI-ask-time\tI-ask-time\tI-ask-time\tB-relation\tI-relation\tO\tO'
	for company in _companys_zh:
		_input = '\t'.join(list(company + '是你们什么时候投资的？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = '\t'.join(_tag_list) + tag_tp
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 谁投资的XX？
	tag_tp = 'B-ask-investor\tB-relation\tI-relation\tO\t'
	for company in _companys_zh:
		_input = '\t'.join(list('谁投资的' + company + '？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 你们有没有投资XX公司？	
	tag_tp = 'B-investor\tI-investor\tO\tO\tO\tB-relation\tI-relation\t'
	for company in _companys_zh:
		_input = '\t'.join(list('你们有没有投资' + company + '？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 你们有投资XX公司吗？	
	tag_tp = 'B-investor\tI-investor\tO\tB-relation\tI-relation\t'
	for company in _companys_zh:
		_input = '\t'.join(list('你们有投资'+ company + '吗？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 你们投资了XX公司吗？	
	tag_tp = 'B-investor\tI-investor\tB-relation\tI-relation\tO\t'
	for company in _companys_zh:
		_input = '\t'.join(list('你们投资了' + company + '吗？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# XX公司被你们投资了吗？	
	tag_tp = '\tO\tB-investor\tI-investor\tB-relation\tI-relation\tO\tO\tO'
	for company in _companys_zh:
		_input = '\t'.join(list(company + '被你们投资了吗？'))
		_tag_list = ['B-company']
		for i in range(1, len(company)):
			_tag_list.append('I-company')
		_tag = '\t'.join(_tag_list) + tag_tp
		# print(_input)
		# print(_tag)
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)
	return dataset
	# with open('data','w', encoding = 'utf-8') as f:
		# f.write('\n\n'.join(dataset))
def build_data_set_en():
	dataset = []
	# XX是不是你们投资的？
	tag_tp = '\tO\tO\tO\tB-investor\tI-investor\tB-relation\tI-relation\tO\tO'
	for company in _companys_en:
		_input = company + '\t' + '\t'.join(list('是不是你们投资的？'))
		_tag_list = ['B-company']
		_tag = '\t'.join(_tag_list) + tag_tp
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# XX是你们投资的吗？
	tag_tp = '\tO\tB-investor\tI-investor\tB-relation\tI-relation\tO\tO\tO'
	for company in _companys_en:
		_input = company + '\t' + '\t'.join(list('是你们投资的吗？'))
		_tag_list = ['B-company']
		_tag = '\t'.join(_tag_list) + tag_tp
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# XX是你们什么时候投资的？
	tag_tp = '\tO\tB-investor\tI-investor\tB-ask-time\tI-ask-time\tI-ask-time\tI-ask-time\tB-relation\tI-relation\tO\tO'
	for company in _companys_en:
		_input = company + '\t' + '\t'.join(list('是你们什么时候投资的？'))
		_tag_list = ['B-company']
		_tag = '\t'.join(_tag_list) + tag_tp
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 谁投资的XX？
	tag_tp = 'B-ask-investor\tB-relation\tI-relation\tO\t'
	for company in _companys_en:
		_input = '\t'.join(list('谁投资的')) + '\t' + company + '\t？'
		_tag_list = ['B-company']
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 你们有没有投资XX公司？	
	tag_tp = 'B-investor\tI-investor\tO\tO\tO\tB-relation\tI-relation\t'
	for company in _companys_en:
		_input = '\t'.join(list('你们有没有投资')) + '\t' + company + '\t？'
		_tag_list = ['B-company']
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 你们有投资XX公司吗？	
	tag_tp = 'B-investor\tI-investor\tO\tB-relation\tI-relation\t'
	for company in _companys_en:
		_input = '\t'.join(list('你们有投资')) + '\t' + '\t吗\t？'
		_tag_list = ['B-company']
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# 你们投资了XX公司吗？	
	tag_tp = 'B-investor\tI-investor\tB-relation\tI-relation\tO\t'
	for company in _companys_en:
		_input = '\t'.join(list('你们投资了')) + '\t' + company + '\t吗\t？'
		_tag_list = ['B-company']
		_tag = tag_tp + '\t'.join(_tag_list) + '\tO\tO'
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)

	# XX公司被你们投资了吗？	
	tag_tp = '\tO\tB-investor\tI-investor\tB-relation\tI-relation\tO\tO\tO'
	for company in _companys_en:
		_input = company + '\t' + '\t'.join(list('被你们投资了吗？'))
		_tag_list = ['B-company']
		_tag = '\t'.join(_tag_list) + tag_tp
		# print(_input)
		# print(_tag)
		assert len(_input.split('\t')) == len(_tag.split('\t'))
		dataset.append(_input + '\n' + _tag)
	return dataset

if __name__ == '__main__':
	dataset = build_data_set_zh()
	# dataset_en = build_data_set_en()
	# pdb.set_trace()
	dataset.extend(build_data_set_en())
	random.shuffle(dataset)
	with open('data', 'w', encoding = 'utf-8') as f:
		f.write('\n'.join(dataset))
	inputs = []
	tags = []
	for line in dataset:
		_input, _tag = line.split('\n')
		inputs.append(_input)
		tags.append(_tag)
	with open('input','w',encoding = 'utf-8') as f:
		f.write('\n'.join(inputs))
	with open('tag', 'w',encoding = 'utf-8') as f:
		f.write('\n'.join(tags))